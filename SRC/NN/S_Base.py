from SRC.NN.IModelBase import IModelBase


class S_Base(IModelBase):
    def inference_discretization_s(self) -> []:
        return self.discretization_s()

    def discretization_s(self) -> []:
        discretization_s = list(self.inference_discretization_feature_s().keys())

        return discretization_s

    def inference_discretization_feature_s(self) -> []:
        return self.discretization_feature_s()

    def discretization_meta_feature_s(self):
        return {
            self.discretization_s()[0]: {
                'ATRr_14'
            }
        }
