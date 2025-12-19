#pragma once

#include <vector>
#include <limits>
#include <tuple>
#include <Core/array.h>

enum class BanditType {
    LOOP = 1,
    LINE = 2
};

class MarkovChain {
public:
    static double beta;

    // Default constructor
    MarkovChain() : type_(BanditType::LINE) {}

    MarkovChain(std::vector<double> done_transitions,
                std::vector<int> done_times,
                std::vector<double> fail_transitions,
                std::vector<int> fail_times,
                BanditType type);

    // Set chain parameters (for default-constructed chains)
    void set(std::vector<double> done_transitions,
             std::vector<int> done_times,
             std::vector<double> fail_transitions,
             std::vector<int> fail_times,
             BanditType type);

    // Equivalent to Python get_subchain(start, end)
    MarkovChain get_subchain(double start, double end) const;

    BanditType get_type() const {return this->type_;};

    // get_gittins_numerator(self, next_layer_numerator=1/(1-beta))
    double get_gittins_numerator(double next_layer_numerator = 1.0 / (1.0 - beta)) const;

    // get_gittins_denominator_aux(self, next_layer_aux=0.0)
    double get_gittins_denominator_aux(double next_layer_aux = 0.0) const;

    struct GittinsResult {
        int stopping_time;
        double numerator;
        double denominator_aux;
    };

    // get_stopping_time_and_gittins_parts(self, state, ...)
    GittinsResult get_stopping_time_and_gittins_parts(
        double state,
        double next_layer_numerator = 1.0 / (1.0 - beta),
        double next_layer_aux = 0.0) const;

private:
    std::vector<double> done_transitions_;
    std::vector<int> done_times_;
    std::vector<double> fail_transitions_;
    std::vector<int> fail_times_;
    std::vector<int> all_times_;
    BanditType type_;

    // Build sorted unique union of done_times and fail_times
    static std::vector<int> compute_all_times(const std::vector<int>& done_times,
                                                 const std::vector<int>& fail_times);

    void normalize_inputs();

    // Internal helpers for the different bandit types
    double get_gittins_numerator_loop(double next_layer_numerator) const;
    double get_gittins_numerator_line(double next_layer_numerator) const;
    double get_gittins_denominator_aux_line(double next_layer_aux) const;
};