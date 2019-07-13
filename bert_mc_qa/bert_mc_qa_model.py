from typing import Dict, Optional, List, Any

import logging
from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel, gelu
import re
import torch
from torch.nn.modules.linear import Linear, Bilinear
from allennlp.modules import TextFieldEmbedder
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn import RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy


@Model.register("bert_mc_qa")
class BertMCQAModel(Model):
    """
    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 top_layer_only: bool = True,
                 bert_weights_model: str = None,
                 per_choice_loss: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 regularizer: Optional[RegularizerApplicator] = None,
                 use_comparative_bert: bool = True,
                 use_bilinear_classifier: bool = False) -> None:
        super().__init__(vocab, regularizer)

        self._use_comparative_bert = use_comparative_bert
        self._use_bilinear_classifier = use_bilinear_classifier

        if bert_weights_model:
            logging.info(f"Loading BERT weights model from {bert_weights_model}")
            bert_model_loaded = load_archive(bert_weights_model)
            self._bert_model = bert_model_loaded.model._bert_model
        else:
            self._bert_model = BertModel.from_pretrained(pretrained_model)

        for param in self._bert_model.parameters():
            param.requires_grad = requires_grad
        #for name, param in self._bert_model.named_parameters():
        #    grad = requires_grad
        #    if layer_freeze_regexes and grad:
        #        grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
        #    param.requires_grad = grad

        bert_config = self._bert_model.config
        self._output_dim = bert_config.hidden_size
        self._dropout = torch.nn.Dropout(bert_config.hidden_dropout_prob)
        self._per_choice_loss = per_choice_loss

        final_output_dim = 1
        if not use_comparative_bert:
            if bert_weights_model and hasattr(bert_model_loaded.model, "_classifier"):
                self._classifier = bert_model_loaded.model._classifier
            else:
                self._classifier = Linear(self._output_dim, final_output_dim)

        else:
            if use_bilinear_classifier:
                self._classifier = Bilinear(self._output_dim, self._output_dim, final_output_dim)
            else:
                self._classifier = Linear(self._output_dim * 2, final_output_dim)
        self._classifier.apply(self._bert_model.init_bert_weights)

        self._all_layers = not top_layer_only
        if self._all_layers:
            if bert_weights_model and hasattr(bert_model_loaded.model, "_scalar_mix") \
                    and bert_model_loaded.model._scalar_mix is not None:
                self._scalar_mix = bert_model_loaded.model._scalar_mix
            else:
                num_layers = bert_config.num_hidden_layers
                initial_scalar_parameters = num_layers * [0.0]
                initial_scalar_parameters[-1] = 5.0  # Starts with most mass on last layer
                self._scalar_mix = ScalarMix(bert_config.num_hidden_layers,
                                             initial_scalar_parameters=initial_scalar_parameters,
                                             do_layer_norm=False)
        else:
            self._scalar_mix = None

        self._accuracy = BooleanAccuracy()
        self._loss = torch.nn.BCEWithLogitsLoss()
        self._debug = -1

    def _extract_last_token_pooled_output(self, encoded_layers, question_mask):
        """
        Extract the output vector for the last token in the sentence -
            similarly to how pooled_output is extracted for us when calling 'bert_model'.
        We need the question mask to find the last actual (non-padding) token
        :return:
        """

        if self._all_layers:
            encoded_layers = encoded_layers[-1]

        # A cool trick to extract the last "True" item in each row
        question_mask = question_mask.squeeze()
        # We already asserted this at batch_size == 1, but why not
        assert question_mask.dim() == 2
        shifted_matrix = question_mask.roll(-1, 1)
        shifted_matrix[:, -1] = 0
        last_item_indices = question_mask - shifted_matrix

        # TODO: This row, for some reason, didn't work as expected, but it is much better then the implementation that follows
        # last_token_tensor = encoded_layers[last_item_indices]

        num_pairs, token_number, hidden_size = encoded_layers.size()
        assert last_item_indices.size() == (num_pairs, token_number)
        # Don't worry, expand doesn't allocate new memory, it simply views the tensor differently
        expanded_last_item_indices = last_item_indices.unsqueeze(2).expand(num_pairs, token_number, hidden_size)
        last_token_tensor = encoded_layers.masked_select(expanded_last_item_indices.byte())
        last_token_tensor = last_token_tensor.reshape(num_pairs, hidden_size)

        pooled_output = self._bert_model.pooler.dense(last_token_tensor)
        pooled_output = self._bert_model.pooler.activation(pooled_output)

        return pooled_output

    def forward(self,
                question: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                choice1_indexes: List[int] = None,
                choice2_indexes: List[int] = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = question['bert']

        # input_ids.size() == (batch_size, num_pairs, max_sentence_length)
        batch_size, num_pairs, _ = question['bert'].size()
        question_mask = (input_ids != 0).long()
        token_type_ids = torch.zeros_like(input_ids)

        # TODO: How to extract last token pooled output if batch size != 1
        assert batch_size == 1

        # Run model
        encoded_layers, first_vectors_pooled_output = self._bert_model(input_ids=util.combine_initial_dims(input_ids),
                                            token_type_ids=util.combine_initial_dims(token_type_ids),
                                            attention_mask=util.combine_initial_dims(question_mask),
                                            output_all_encoded_layers=self._all_layers)

        if self._use_comparative_bert:
            last_vectors_pooled_output = self._extract_last_token_pooled_output(encoded_layers, question_mask)
        else:
            last_vectors_pooled_output = None
        if self._all_layers:
            mixed_layer = self._scalar_mix(encoded_layers, question_mask)
            first_vectors_pooled_output = self._bert_model.pooler(mixed_layer)

        # Apply dropout
        first_vectors_pooled_output = self._dropout(first_vectors_pooled_output)
        if self._use_comparative_bert:
            last_vectors_pooled_output = self._dropout(last_vectors_pooled_output)

        # Classify
        if not self._use_comparative_bert:
            pair_label_logits = self._classifier(first_vectors_pooled_output)
        else:
            if self._use_bilinear_classifier:
                pair_label_logits = self._classifier(first_vectors_pooled_output, last_vectors_pooled_output)
            else:
                all_pooled_output = torch.cat((first_vectors_pooled_output, last_vectors_pooled_output), 1)
                pair_label_logits = self._classifier(all_pooled_output)

        pair_label_logits_flat = pair_label_logits.squeeze(1)
        pair_label_logits = pair_label_logits.view(-1, num_pairs)

        output_dict = {}
        output_dict['pair_label_logits'] = pair_label_logits
        output_dict['choice1_indexes'] = choice1_indexes
        output_dict['choice2_indexes'] = choice2_indexes

        output_dict['pair_label_probs'] = torch.sigmoid(pair_label_logits_flat).view(-1, num_pairs)

        if label is not None:
            label = label.unsqueeze(1)
            label = label.expand(-1, num_pairs)
            relevant_pairs = (choice1_indexes == label) | (choice2_indexes == label)
            relevant_logits = pair_label_logits[relevant_pairs]
            choice1_is_the_label = (choice1_indexes == label)[relevant_pairs]
            # choice1_is_the_label = choice1_is_the_label.type_as(relevant_logits)

            loss = self._loss(relevant_logits, choice1_is_the_label.float())
            self._accuracy(relevant_logits >= 0.5, choice1_is_the_label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'EM': self._accuracy.get_metric(reset),
        }

