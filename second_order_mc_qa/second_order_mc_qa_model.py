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


@Model.register("second_order_mc_qa")
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

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        self._debug = -1

    def forward(self,
                pair_probs: torch.FloatTensor,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1

        # input_ids.size() == (batch_size, num_pairs, max_sentence_length)
        batch_size, num_pairs = pair_probs.size()

        #TODO: assert no -1's in pair_probs_field

        # TODO: How to extract last token pooled output if batch size != 1
        assert batch_size == 1

        # TODO: Apply dropout
        # Run model
        choice_logits = self._layer_2(self._layer_1_activation(self._layer_1(pair_probs)))
        
        output_dict = {}
        output_dict['choice_logits'] = choice_logits
        output_dict['choice_probs'] = torch.softmax(choice_logits, 1)
        output_dict['predicted_choice'] = torch.argmax(choice_logits, 1)

        if label is not None:
            loss = self._loss(choice_logits, label)
            self._accuracy(choice_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'EM': self._accuracy.get_metric(reset),
        }

