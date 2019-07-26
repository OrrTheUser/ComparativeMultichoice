from typing import Dict, Tuple, List

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from bert_mc_qa.elastic_logger import ElasticLogger
import numpy as np

NUMBER_OF_CHOICES = 5

@Predictor.register('bert_mc_qa')
class MCQAPredictor(Predictor):
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        if 'id' in json_dict:
            item_id = json_dict['id']
        else:
            item_id = ''
        if 'username' in json_dict:
            username = json_dict['username']
        else:
            username = ''
        question = json_dict['question']
        question_text = question['stem']

        choice_dict_list = question['choices']
        choice_dict_list.sort(key=lambda x: x['label'])
        choice_list = [x['text'] for x in choice_dict_list]

        instance = self._dataset_reader.text_to_instance(item_id, question_text, choice_list)
        predictions = self.predict_instance(instance)

        zipped = list(zip(predictions['choice1_indexes'], predictions['choice2_indexes'], predictions['pair_label_probs']))
        pair_to_prob_dict = {(i[0], i[1]): i[2] for i in zipped}
        pair_to_prob_dict_str = {str((i[0], i[1])): i[2] for i in zipped}

        choice_prob_list = calculate_choice_probs(pair_to_prob_dict)

        predicted_answer_index = np.argmax(choice_prob_list)
        # sorted_predictions = sorted(choice_prob_list, reverse = True)
        # if sorted_predictions[0] - sorted_predictions[1] < 0.03:
        #    predicted_answer = "I'm not sure"

        predicted_answer_label = choice_dict_list[predicted_answer_index]['label']

        example = {'username': username, 'pred_answer': predicted_answer_label,'question': question,
                   'predictions': choice_prob_list, 'choices': choice_dict_list, 'id': item_id,
                   'pair_prob_dict': pair_to_prob_dict_str}

        if 'answerKey' in json_dict:
            example['gold_answer_key_for_comparisons_only'] = json_dict['answerKey']

        if 'write_log' in json_dict:
            ElasticLogger().write_log('INFO', 'example', context_dict=example)

        return example


def calculate_choice_probs(pair_to_prob_dict: Dict[Tuple[int, int], float]) -> List[float]:
    pair_probs_matrix = np.zeros((NUMBER_OF_CHOICES, NUMBER_OF_CHOICES))
    for pair in pair_to_prob_dict:
        pair_probs_matrix[pair] = pair_to_prob_dict[pair]

    # Transpose, so that a (row -> col) edge represents "How much better is 'col' than 'row'"
    pair_probs_matrix = pair_probs_matrix.transpose()

    # Do we want symmetry (in a ones-complement sense)
    should_be_symmetric = True
    if should_be_symmetric:
        complement_weights = (1 - pair_probs_matrix).transpose() - np.eye(NUMBER_OF_CHOICES)
        pair_probs_matrix = (pair_probs_matrix + complement_weights) / 2

    # Add column-averages as diagonals (to strengthen good values)
    col_sums = pair_probs_matrix.sum(axis=0) / (NUMBER_OF_CHOICES - 1)
    pair_probs_matrix = np.eye(NUMBER_OF_CHOICES) * col_sums + pair_probs_matrix

    # Normalize rows, so that we have graph probability matrix
    row_sums = pair_probs_matrix.sum(axis=1)
    pair_probs_matrix = pair_probs_matrix / row_sums[:, np.newaxis]

    # Find the stationary probability vector for this (markov chain) graph
    prob_vector = solve_stationary(pair_probs_matrix)
    return list(prob_vector.flat)

def solve_stationary( A ):
    """ x = xA where x is the answer
    x - xA = 0
    x( I - A ) = 0 and sum(x) = 1
    """
    n = A.shape[0]
    a = np.eye( n ) - A
    a = np.vstack( (a.T, np.ones( n )) )
    b = np.matrix( [0] * n + [ 1 ] ).T
    return np.linalg.lstsq( a, b )[0]
