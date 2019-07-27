from typing import Dict, List, Any
import itertools
import json
import logging
import numpy
import re
import random
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, Field, TextField, LabelField, MultiLabelField, SequenceField
from allennlp.data.fields import ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

INDEX_TO_CHOICE_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
CHOICE_LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

@DatasetReader.register("second_order_mc_qa")
class SecondOrderMCQAReader(DatasetReader):
    """
    Reads a file from the AllenAI-V1-Feb2018 dataset in Json format.  This data is
    formatted as jsonl, one json-formatted instance per line.  An example of the json in the data is:
        {"id":"MCAS_2000_4_6",
        "question":{"stem":"Which technology was developed most recently?",
            "choices":[
                {"text":"cellular telephone","label":"A"},
                {"text":"television","label":"B"},
                {"text":"refrigerator","label":"C"},
                {"text":"airplane","label":"D"}
            ]},
        "answerKey":"A"
        }
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 num_choices: int = 5,
                 random_seed: int = 0) -> None:
        super().__init__()

        self._num_choices = num_choices
        self._random_seed = random_seed

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        debug = 5

        with open(file_path, 'r') as data_file:
            logger.info("Reading QA instances from jsonl dataset at: %s", file_path)
            instances = []
            for line in data_file:
                debug -= 1
                item_json = json.loads(line.strip())

                if debug > 0:
                    logger.info(item_json)

                item_id = item_json["id"]
                answer_id = None
                if 'gold_answer_key_for_comparisons_only' in item_json:
                    answer_key = item_json['gold_answer_key_for_comparisons_only']
                    answer_id = CHOICE_LETTER_TO_INDEX[answer_key]
                pair_prob_dict_serialized = item_json['pair_prob_dict']

                instances.append(self.text_to_instance(
                    item_id,
                    pair_prob_dict_serialized,
                    answer_id,
                    debug))

            random.seed(self._random_seed)
            random.shuffle(instances)
            for instance in instances:
                yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         pair_prob_dict_serialized: Dict[str, float],
                         answer_id: int = None,
                         debug: int = -1) -> Instance:

        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        pair_prob_dict = {}
        for key in pair_prob_dict_serialized:
            key_as_tuple = tuple(map(int, key[1:-1].split(',')))
            pair_prob_dict[key_as_tuple] = pair_prob_dict_serialized[key]

        pair_probs_matrix = numpy.zeros((self._num_choices, self._num_choices))
        for pair in pair_prob_dict:
            pair_probs_matrix[pair] = pair_prob_dict[pair]

        real_indexes = numpy.ones((self._num_choices, self._num_choices), dtype=numpy.byte) - \
            numpy.eye(self._num_choices, dtype=numpy.byte)
        pair_probs = pair_probs_matrix[real_indexes == 1]

        assert pair_probs.shape == (self._num_choices * (self._num_choices - 1),)

        fields['pair_probs'] = ArrayField(pair_probs, padding_value=-1)

        if answer_id is not None:
            fields['label'] = LabelField(answer_id, skip_indexing=True)

        metadata = {
            "id": item_id,
            "pair_probs_dict_serialized": pair_prob_dict_serialized,
            "correct_answer_index": answer_id,
        }

        if debug > 0:
            logger.info(f"answer_id = {answer_id}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)