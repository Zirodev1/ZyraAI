// include/personalization/personality_manager.h

#ifndef ZYRAAI_PERSONALITY_MANAGER_H
#define ZYRAAI_PERSONALITY_MANAGER_H

#include <string>
#include <unordered_map>

namespace zyraai {
    class PersonalityManager {
    public:
        PersonalityManager();
        ~PersonalityManager();

        void addPersonality(const std::string& name, const std::string& behaviorScript);
        void setActivePersonality(const std::string& name);
        std::string getResponse(const std::string& input);

    private:
        std::unordered_map<std::string, std::string> personalities;
        std::string activePersonality;
    };
}

#endif
