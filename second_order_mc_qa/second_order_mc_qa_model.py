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
                 num_choices: int,
                 hidden_size: int = -1,
                 use_relu: bool = True) -> None:
        super().__init__(None, None)

        self._num_choices = num_choices
        num_choices_squared = num_choices * num_choices
        if hidden_size == -1:
            hidden_size = num_choices_squared * num_choices_squared
        self._hidden_size = hidden_size

        self._layer_1 = Linear(num_choices_squared, hidden_size)
        if use_relu:
            self._layer_1_activation = torch.nn.LeakyReLU()
        else:
            self._layer_1_activation = torch.nn.Tanh()
        self._layer_2 = Linear(hidden_size, num_choices)
        self._layer_2_activation = torch.nn.Softmax()

        self._accuracy = BooleanAccuracy()
        self._loss = torch.nn.BCEWithLogitsLoss()

        self._debug = -1

    def forward(self,
                pair_probs: torch.FloatTensor,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1

        # input_ids.size() == (batch_size, num_pairs, max_sentence_length)
        batch_size, num_pairs, _ = pair_probs.size()

        #TODO: assert no -1's in pair_probs_field

        # TODO: How to extract last token pooled output if batch size != 1
        assert batch_size == 1

        # TODO: Apply dropout
        # Run model
        choice_logits = self._layer_2(self._layer_1_activation(self._layer_1(pair_probs)))

        import ipdb
        ipdb.set_trace()



        pair_label_logits = pair_label_logits.view(-1, num_pairs)

        pair_label_probs = torch.sigmoid(pair_label_logits)
        pair_label_probs_flat = pair_label_probs.squeeze(1)

        output_dict = {}
        output_dict['pair_label_logits'] = pair_label_logits
        output_dict['choice1_indexes'] = choice1_indexes
        output_dict['choice2_indexes'] = choice2_indexes

        output_dict['pair_label_probs'] = pair_label_probs_flat.view(-1, num_pairs)

        if label is not None:
            label = label.unsqueeze(1)
            label = label.expand(-1, num_pairs)
            relevant_pairs = (choice1_indexes == label) | (choice2_indexes == label)
            relevant_probs = pair_label_probs[relevant_pairs]
            choice1_is_the_label = (choice1_indexes == label)[relevant_pairs]
            # choice1_is_the_label = choice1_is_the_label.type_as(relevant_logits)

            loss = self._loss(relevant_probs, choice1_is_the_label.float())
            self._accuracy(relevant_probs >= 0.5, choice1_is_the_label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'EM': self._accuracy.get_metric(reset),
        }

