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


@DatasetReader.register("bert_mc_qa")
class BertMCQAReader(DatasetReader):
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
                 pretrained_model: str,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_pieces: int = 512,
                 num_choices: int = 5,
                 answer_only: bool = False,
                 restrict_num_choices: int = None,
                 ignore_context: bool = False,
                 sample: int = -1,
                 random_seed: int = 0) -> None:
        super().__init__()

        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        lower_case = not '-cased' in pretrained_model
        self._word_splitter = BertBasicWordSplitter(do_lower_case=lower_case)
        self._max_pieces = max_pieces
        self._sample = sample
        self._num_choices = num_choices
        self._answer_only = answer_only
        self._restrict_num_choices = restrict_num_choices
        self._ignore_context = ignore_context
        self._random_seed = random_seed

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = self._sample + 1
        debug = 5

        with open(file_path, 'r') as data_file:
            logger.info("Reading QA instances from jsonl dataset at: %s", file_path)
            instances = []
            for line in data_file:
                counter -= 1
                debug -= 1
                if counter == 0:
                    break
                item_json = json.loads(line.strip())

                if debug > 0:
                    logger.info(item_json)

                item_id = item_json["id"]
                context = item_json.get("para")
                if self._ignore_context:
                    context = None
                question_text = item_json["question"]["stem"]

                if self._answer_only:
                    question_text = ""

                choice_label_to_id = {}
                choice_text_list = []
                choice_context_list = []

                any_correct = False
                choice_id_correction = 0

                for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
                    if self._restrict_num_choices and len(choice_text_list) == self._restrict_num_choices:
                        if not any_correct:
                            choice_text_list.pop(-1)
                            choice_context_list.pop(-1)
                            choice_id_correction += 1
                        else:
                            break

                    choice_label = choice_item["label"]
                    choice_label_to_id[choice_label] = choice_id - choice_id_correction

                    choice_text = choice_item["text"]
                    choice_context = choice_item.get("para")
                    if self._ignore_context:
                        choice_context = None

                    choice_text_list.append(choice_text)
                    choice_context_list.append(choice_context)

                    if item_json.get('answerKey') == choice_label:
                        if any_correct:
                            raise ValueError("More than one correct answer found for {item_json}!")
                        any_correct = True

                    if self._restrict_num_choices \
                            and len(choice_text_list) == self._restrict_num_choices \
                            and not any_correct:
                        continue


                if not any_correct and 'answerKey' in item_json:
                    raise ValueError("No correct answer found for {item_json}!")

                answer_id = choice_label_to_id[item_json["answerKey"]]
                # Pad choices with empty strings if not right number
                if len(choice_text_list) != self._num_choices:
                    choice_text_list = (choice_text_list + self._num_choices * [''])[:self._num_choices]
                    choice_context_list = (choice_context_list + self._num_choices * [None])[:self._num_choices]
                    if answer_id >= self._num_choices:
                        logging.warning(f"Skipping question with more than {self._num_choices} answers: {item_json}")
                        continue

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
        #TODO: remove all segment_ids references in this file - they are never referenced (the wordpiece indexer implements "bert-type-ids")
        segment_ids_fields = []
        pair_tokens_list = []
        choice1_index_fields = []
        choice2_index_fields = []

        for index1, index2 in itertools.combinations(range(len(choice_list)), 2):
            choice1, choice2 = (choice_list[index1], choice_list[index2])
            # TODO: What to do if contexts are not none?
            assert context is None and all(map(lambda x: x is None, choice_context_list))
            pair_tokens, segment_ids = self.bert_features_from_q_2a(question, choice1, choice2)
            pair_field = TextField(pair_tokens, self._token_indexers)
            segment_ids_field = ArrayField(segment_ids, dtype=numpy.int)
            choice1_index_field = LabelField(index1, skip_indexing=True)
            choice2_index_field = LabelField(index2, skip_indexing=True)
            pair_fields.append(pair_field)
            pair_tokens_list.append(pair_tokens)
            segment_ids_fields.append(segment_ids_field)
            choice1_index_fields.append(choice1_index_field)
            choice2_index_fields.append(choice2_index_field)
            if debug > 0:
                logger.info(f"qa_tokens = {pair_tokens}")
                logger.info(f"segment_ids = {segment_ids}")

        fields['question'] = ListField(pair_fields)
        fields['segment_ids'] = ListField(segment_ids_fields)
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
        segment_ids = list(itertools.repeat(1, len(choice1_tokens) + 2)) +\
                      list(itertools.repeat(0, len(question_tokens))) + \
                      list(itertools.repeat(1, len(choice2_tokens) + 2))
        segment_ids = numpy.array(segment_ids)
        return tokens, segment_ids
