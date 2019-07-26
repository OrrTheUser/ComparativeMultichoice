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
                answer_key = None
                if 'answerKey' in item_json:
                    answer_key = item_json['answerKey']
                pair_prob_dict_serialized = item_json['pair_prob_dict']

                pair_prob_dict = {}
                for key in pair_prob_dict_serialized:
                    key_as_tuple = tuple(map(int, key[1:-1].split(',')))
                    pair_prob_dict[key_as_tuple] = pair_prob_dict_serialized[key]

                pair_probs_matrix = numpy.zeros((self._num_choices, self._num_choices))
                for pair in pair_prob_dict:
                    pair_probs_matrix[pair] = pair_prob_dict[pair]

                pair_probs = pair_probs_matrix.flatten()

                instances.append(self.text_to_instance(
                    item_id,
                    question_text,
                    choice_text_list,
                    answer_id,
                    context,
                    choice_context_list,
                    debug))

            random.seed(self._random_seed)
            random.shuffle(instances)
            for instance in instances:
                yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         question: str,
                         choice_list: List[str],
                         answer_id: int = None,
                         context: str = None,
                         choice_context_list: List[str] = None,
                         debug: int = -1) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        pair_fields = []
        pair_tokens_list = []
        choice1_index_fields = []
        choice2_index_fields = []

        for index1, index2 in itertools.permutations(range(len(choice_list)), 2):
            choice1, choice2 = (choice_list[index1], choice_list[index2])
            # TODO: What to do if contexts are not none?
            assert context is None
            if choice_context_list is not None:
                assert all(map(lambda x: x is None, choice_context_list))
            pair_tokens = self.bert_features_from_q_2a(question, choice1, choice2)
            pair_field = TextField(pair_tokens, self._token_indexers)
            choice1_index_field = LabelField(index1, skip_indexing=True)
            choice2_index_field = LabelField(index2, skip_indexing=True)
            pair_fields.append(pair_field)
            pair_tokens_list.append(pair_tokens)
            choice1_index_fields.append(choice1_index_field)
            choice2_index_fields.append(choice2_index_field)
            if debug > 0:
                logger.info(f"qa_tokens = {pair_tokens}")

        fields['question'] = ListField(pair_fields)
        fields['choice1_indexes'] = ListField(choice1_index_fields)
        fields['choice2_indexes'] = ListField(choice2_index_fields)

        if answer_id is not None:
            fields['label'] = LabelField(answer_id, skip_indexing=True)

        metadata = {
            "id": item_id,
            "question_text": question,
            "choice_text_list": choice_list,
            "correct_answer_index": answer_id,
            "question_tokens_list": pair_tokens_list
        }

        if debug > 0:
            logger.info(f"answer_id = {answer_id}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    @staticmethod
    def _truncate_tokens(tokens_a, tokens_b, tokens_c, max_length):
        """
        Truncate 'a' from the start, 'b' form the start, and 'c' from the end until total is less than max_length.
        At each step, truncate the longest one
        """
        while len(tokens_a) + len(tokens_b) + len(tokens_c) > max_length:
            reduction_candidate = numpy.argmax(len(tokens_a), len(tokens_b), len(tokens_c))
            if reduction_candidate == 0:
                # 'a' is the longest
                tokens_a.pop(0)
            elif reduction_candidate == 1:
                # 'b' is the longest
                tokens_b.pop(0)
            else:
                # 'c' is the longest
                tokens_c.pop()
        return tokens_a, tokens_b, tokens_c

    def bert_features_from_q_2a(self, question: str, answer1: str, answer2: str, context: str = None):
        #TODO: What should we do if context is not None (where to append it?)
        assert context is None

        sep_token = Token("[SEP]")
        question_tokens = self._word_splitter.split_words(question)

        choice1_tokens = self._word_splitter.split_words(answer1)
        choice2_tokens = self._word_splitter.split_words(answer2)
        question_tokens, choice1_tokens, choice2_tokens = self._truncate_tokens(question_tokens, choice1_tokens,
                                                                                choice2_tokens, self._max_pieces - 2)

        tokens = choice1_tokens + [sep_token] + question_tokens + [sep_token] + choice2_tokens
        return tokens
