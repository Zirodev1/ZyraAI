// src/personization/personality_manager.cpp

#include "personalization/personality_manager.h"
#include <iostream>

namespace zyraai {
    PersonalityManager::PersonalityManager(){
        std::cout << "Personality Manager Initailized" << std::endl;
    }

    PersonalityManager::~PersonalityManager() {
        std::cout << "Personality Mangaer Destroyed" << std::endl;
    }

    void PersonalityManager::addPersonality(const std::string& name, const std::string& behaviorScript) {
        personalities[name] = behaviorScript;
        std::cout << "Added personality: " << name << std::endl;
    }

    void PersonalityManager::setActivePersonality(const std::string& name) {
        if (personalities.find(name) != personalities.end()) {
            activePersonality = name;
            std::cout << "Active personality set to: " << name << std::endl;
        } else {
            std::cerr << "Personality not found: " << name << std::endl;
        }
    }

    std::string PersonalityManager::getResponse(const std::string& input) {
        if(personalities.find(activePersonality) != personalities.end()) {
            // placeholder for processing behavior script and generating response
            return "Response based on " + activePersonality +" personality to input: " + input;
        } else {
            return "No active personality set.";
        }
    }
}