# pragma once
#include <Core/array.h>
#include "../Core/util.h"

namespace rai {

    struct TaskPlan;

    struct Action;

    double planDistance(const TaskPlan& a, const TaskPlan& b);

    double actionDistance(const Action& a, const Action& b);

    struct Action {
        rai::String verb;
        rai::Array<rai::String> objects;

        Action();
        Action(const rai::String& str);
        Action(const Action& other);
        Action(const std::string& str);
        Action(rai::String verb, rai::Array<rai::String> objects);
        ~Action();

        std::string toString() const;

        Action& operator=(const Action& other);
        bool operator==(const Action& other) const;
        bool operator!=(const Action& other) const;
        friend std::ostream& operator<<(std::ostream& os, const Action& a);
    };

    struct TaskPlan {
        rai::Array<Action> actions;
        bool empty = false;

        TaskPlan(const rai::Array<Action>& actions);
        TaskPlan(const rai::String& planString);
        TaskPlan(const std::string& planString);
        TaskPlan(const TaskPlan& other);
        TaskPlan(const Array<StringA> actionSequence);
        TaskPlan();
        ~TaskPlan();

        std::string toString() const;

        TaskPlan& operator=(const TaskPlan& other);
        bool operator==(const TaskPlan& other) const;
        bool operator!=(const TaskPlan& other) const;
        friend std::ostream& operator<<(std::ostream& os, const TaskPlan& tp);
    };
}