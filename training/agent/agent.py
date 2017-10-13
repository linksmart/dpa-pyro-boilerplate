"""
Training / Prediction Agent
"""

import numpy as np
import logging


logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__file__)


class Agent(object):

    def build(self, classifier):
        logger.info("agent.build: %s" % classifier)

    def learn(self, datapoint):
        logger.info("agent.learn: %s" % datapoint)

    def predict(self, datapoint):
        logger.info("agent.predict: %s" % datapoint)

        return []

    def batchLearn(self, datapoints):
        logger.info("agent.batchLearn: %s" % datapoints)

    def batchPredict(self, datapoints):
        logger.info("agent.batchPredict: %s" % datapoints)
        return np.zeros(len(datapoints)).astype(int).tolist()

    def destroy(self):
        logger.info("agent.destroy")

    def exportModel(self):
        raise NotImplementedError
        # Zip tmp directory and return binaries?
        # http://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory
        # pmml: https://github.com/alex-pirozhenko/sklearn-pmml

    def importModel(self):
        raise NotImplementedError

   