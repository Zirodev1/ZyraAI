// include/data/data_handler.h

#ifndef ZYRAAI_DATA_HANDLER_H
#define ZYRAAI_DATA_HANDLER_H

#include <string>
#include <vector>

namespace zyraai {
    class DataHandler {
        public:
            DataHandler();
            ~DataHandler();

            void loadData(const std::string& filePath);
            void processData();
            void saveData(const std::string& filePath);
            std::vector<std::string> getData() const;

        private:
            std::vector<std::string> data;
    };
}

#endif // zyraAI_data_handler