// include/model/zyraAI_model.h

#ifndef ZYRAAI_MODEL_H
#define ZYRAAI_MODEL_H

namespace zyraai {
    class ZyraAIModel {
    public:
        ZyraAIModel();
        ~ZyraAIModel();

        void train();
        void predict();
    };
}

#endif // ZYRAAI_MODEL_H
