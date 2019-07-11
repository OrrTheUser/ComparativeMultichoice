from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from bert_mc_qa.elastic_logger import ElasticLogger
import numpy as np

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

        # Our model was usually trained on 5 answers, so we will need to filter out 2 answers..
        choice_list = [json_dict['choice1'], json_dict['choice2'], json_dict['choice3']]
        instance = self._dataset_reader.text_to_instance(item_id, question, choice_list)
        predictions = self.predict_instance(instance)
        predicted_answer = choice_list[np.argmax(predictions['label_probs'])]
        sorted_predictions = sorted(predictions['label_probs'], reverse = True)
        if sorted_predictions[0] - sorted_predictions[1] < 0.03:
            predicted_answer = "I'm not sure"

        example = {'username':username, 'pred_answer':predicted_answer,'question': question, 'predictions': predictions['label_probs'][0:3], \
                   'predictions_logits': predictions['label_logits'][0:3],'model':json_dict['model'], \
                   'choice1':json_dict['choice1'],'choice2':json_dict['choice2'],'choice3':json_dict['choice3']}

        if 'write_log' in json_dict:
            ElasticLogger().write_log('INFO', 'example', context_dict=example)

        return example
