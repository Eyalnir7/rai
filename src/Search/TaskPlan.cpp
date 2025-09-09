#include "Search/TaskPlan.h"
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <string>
#include <set>
#include <utility>
#include <cstddef>

namespace {
    template <class Range1,
          class Key = typename Range1::value_type,
          class Hash = std::hash<Key>,
          class Eq   = std::equal_to<Key>>
    std::pair<std::size_t, std::size_t> multiset_inter_union_counts(const Range1& A, const Range1& B, Hash hasher = Hash{}, Eq eq = Eq{}) {

        std::unordered_map<Key, std::size_t, Hash, Eq> freq(0, hasher, eq);

        std::size_t sizeA = 0, sizeB = 0, inter = 0;

        // Count A
        for (const auto& x : A) {
            ++freq[x];
            ++sizeA;
        }

        // Walk B and count intersection by consuming A's counts
        for (const auto& y : B) {
            ++sizeB;
            auto it = freq.find(y);
            if (it != freq.end() && it->second > 0) {
                --(it->second);
                ++inter;
            }
        }

        std::size_t uni = sizeA + sizeB - inter;
        if (uni <= 0)
        {
            throw std::runtime_error("Invalid union size");
        }
        return {inter, uni};
    }
}

namespace rai {

    std::multiset<std::string> planAsMultiSet(const TaskPlan& plan) {
        std::multiset<std::string> result;
        for (const auto& action : plan.actions) {
            result.insert(action.verb.p);
            for (uint i = 0; i < action.objects.N; i++) {
                result.insert(action.objects(i).p);
            }
        }
        return result;
    }

    // create a function similar to one above but for Action
    std::multiset<std::string> actionAsMultiSet(const Action& action) {
        std::multiset<std::string> result;
        result.insert(action.verb.p);
        for (uint i = 0; i < action.objects.N; i++) {
            result.insert(action.objects(i).p);
        }
        return result;
    }

    //Create a function like planDistance but actionDistance
    double actionDistance(const Action& a, const Action& b) {
        if(a == b) return -1;
        auto [inter, uni] = multiset_inter_union_counts(actionAsMultiSet(a), actionAsMultiSet(b));
        return 1 - static_cast<double>(inter) / uni;
    }

    // Utility function to calculate plan distance
    double planDistance(const TaskPlan& a, const TaskPlan& b) {
        if(a == b) return -1;
        auto [inter, uni] = multiset_inter_union_counts(planAsMultiSet(a), planAsMultiSet(b));
        return 1 - static_cast<double>(inter) / uni;
    }

    // Action class implementation
    Action::Action(const rai::String& str) {
        // Parse action string in format "(verb obj1 obj2 ...)"
        std::string s = str.p;
        
        // Trim whitespace from the ends of the string
        size_t start = s.find_first_not_of(" \t\r\n");
        size_t end = s.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end != std::string::npos) {
            s = s.substr(start, end - start + 1);
        }
        
        // Check if string starts and ends with parentheses
        if (s.front() == '(' && s.back() == ')') {
            // Remove parentheses and parse content
            std::string content = s.substr(1, s.length() - 2);
            std::stringstream ss(content);
            std::string token;
            bool isFirst = true;
            
            while (ss >> token) {
                if (isFirst) {
                    verb = STRING(token);
                    isFirst = false;
                } else {
                    objects.append(STRING(token));
                }
            }
        } else {
            // Fallback: parse as space-separated without parentheses
            std::stringstream ss(s);
            std::string token;
            bool isFirst = true;
            
            while (ss >> token) {
                if (isFirst) {
                    verb = STRING(token);
                    isFirst = false;
                } else {
                    objects.append(STRING(token));
                }
            }
        }
    }

    Action::Action(const std::string& str) 
        : Action(rai::String(STRING(str))) {
    }

    Action::Action(const Action& other) 
        : verb(other.verb), objects(other.objects) {
    }

    Action::Action(rai::String verb, rai::Array<rai::String> objects) 
        : verb(verb), objects(objects) {
    }

    Action::~Action() {
    }

    Action::Action() : verb(""), objects() {
    }

    std::string Action::toString() const {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }

    Action& Action::operator=(const Action& other) {
        if (this != &other) {
            verb = other.verb;
            objects = other.objects;
        }
        return *this;
    }

    bool Action::operator==(const Action& other) const {
        // Check if verbs are equal
        if (verb != other.verb) {
            return false;
        }
        
        // Check if object arrays have the same size
        if (objects.N != other.objects.N) {
            return false;
        }
        
        // Check if all objects are equal in the same order
        for (uint i = 0; i < objects.N; i++) {
            if (objects(i) != other.objects(i)) {
                return false;
            }
        }
        
        return true;
    }

    bool Action::operator!=(const Action& other) const {
        return !(*this == other);
    }

    std::ostream& operator<<(std::ostream& os, const Action& a) {
        os << "(" << a.verb.p;
        for (uint i = 0; i < a.objects.N; i++) {
            os << " " << a.objects(i).p;
        }
        os << ")";
        return os;
    }

    // TaskPlan class implementation
    TaskPlan::TaskPlan(const rai::Array<Action>& actions) 
        : actions(actions) {
            empty = false;
    }

    TaskPlan::TaskPlan(){
        actions = rai::Array<Action>();
        empty = true;
    }

    TaskPlan::TaskPlan(const Array<StringA> actionSequence) {
        actions.clear();
        for (const auto& actionArr : actionSequence) {
            if(actionArr.N > 0) {
                rai::String verb = actionArr(0);
                rai::Array<rai::String> objects;
                for (uint i = 1; i < actionArr.N; i++) {
                    objects.append(actionArr(i));
                }
                actions.append(Action(verb, objects));
            }
        }
        empty = actions.N == 0;
    }

    TaskPlan::TaskPlan(const rai::String& planString) {
        // Parse plan string with actions in format "(verb obj1 obj2 ...)"
        std::string s = planString.p;
        
        // Find all actions enclosed in parentheses
        size_t pos = 0;
        while (pos < s.length()) {
            // Find the start of next action
            size_t start = s.find('(', pos);
            if (start == std::string::npos) break;
            
            // Find the matching closing parenthesis
            size_t end = s.find(')', start);
            if (end == std::string::npos) break;
            
            // Extract the action string including parentheses
            std::string actionStr = s.substr(start, end - start + 1);
            
            // Create action and add to plan
            actions.append(Action(rai::String(STRING(actionStr))));
            
            // Move past this action
            pos = end + 1;
        }
        empty = false;
    }

    TaskPlan::TaskPlan(const std::string& planString) 
        : TaskPlan(rai::String(STRING(planString))) {
    }

    TaskPlan::TaskPlan(const TaskPlan& other) 
        : actions(other.actions), empty(other.empty) {
    }

    TaskPlan::~TaskPlan() {
    }

    std::string TaskPlan::toString() const {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }

    TaskPlan& TaskPlan::operator=(const TaskPlan& other) {
        if (this != &other) {
            actions = other.actions;
            empty = other.empty; 
        }
        return *this;
    }

    bool TaskPlan::operator==(const TaskPlan& other) const {
        // Check if empty flags are equal
        if (empty != other.empty) {
            return false;
        }
        
        // Check if action arrays have the same size
        if (actions.N != other.actions.N) {
            return false;
        }
        
        // Check if all actions are equal in the same order
        for (uint i = 0; i < actions.N; i++) {
            if (actions(i) != other.actions(i)) {
                return false;
            }
        }
        
        return true;
    }

    bool TaskPlan::operator!=(const TaskPlan& other) const {
        return !(*this == other);
    }

    std::ostream& operator<<(std::ostream& os, const TaskPlan& tp) {
        for (uint i = 0; i < tp.actions.N; i++) {
            if (i > 0) os << " ";
            os << tp.actions(i);
        }
        return os;
    }

} // namespace rai
