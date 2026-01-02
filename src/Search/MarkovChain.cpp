#include "MarkovChain.h"
#include <Core/util.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <utility>   // std::move
#include <limits>    // infinity, quiet_NaN

// Define and initialize static member from config
double MarkovChain::beta = rai::getParameter<double>("Bandit/beta", 0.9999);

// ======== Public methods ========

MarkovChain::MarkovChain(std::vector<double> done_transitions,
                         std::vector<int> done_times,
                         std::vector<double> fail_transitions,
                         std::vector<int> fail_times,
                         BanditType type)
    : done_transitions_(std::move(done_transitions)),
      done_times_(std::move(done_times)),
      fail_transitions_(std::move(fail_transitions)),
      fail_times_(std::move(fail_times)),
      type_(type)
{
    normalize_inputs();
}

void MarkovChain::set(std::vector<double> done_transitions,
                      std::vector<int> done_times,
                      std::vector<double> fail_transitions,
                      std::vector<int> fail_times,
                      BanditType type) {
    done_transitions_ = std::move(done_transitions);
    done_times_ = std::move(done_times);
    fail_transitions_ = std::move(fail_transitions);
    fail_times_ = std::move(fail_times);
    type_ = type;
    normalize_inputs();
}

MarkovChain MarkovChain::get_subchain(double start, double end) const {
    std::vector<int> done_t, fail_t;
    std::vector<double> done_p, fail_p;

    // Determine the shift amount: if start is -infinity (< 0), shift by 0, otherwise shift by start
    const int shift = (start < 0) ? 0 : static_cast<int>(start);

    done_t.reserve(done_times_.size());
    done_p.reserve(done_transitions_.size());
    for (std::size_t i = 0; i < done_times_.size(); ++i) {
        const int t = done_times_[i];
        if (t >= start && t <= end) {
            done_t.push_back(t - shift);
            done_p.push_back(done_transitions_[i]);
        }
    }

    fail_t.reserve(fail_times_.size());
    fail_p.reserve(fail_transitions_.size());
    for (std::size_t i = 0; i < fail_times_.size(); ++i) {
        const int t = fail_times_[i];
        if (t >= start && t <= end) {
            fail_t.push_back(t - shift);
            fail_p.push_back(fail_transitions_[i]);
        }
    }

    return MarkovChain(done_p, done_t, fail_p, fail_t, type_);
}

double MarkovChain::get_gittins_numerator(double next_layer_numerator) const {
    if (type_ == BanditType::LOOP) {
        return get_gittins_numerator_loop(next_layer_numerator);
    }
    return get_gittins_numerator_line(next_layer_numerator);
}

double MarkovChain::get_gittins_denominator_aux(double next_layer_aux) const {
    if (type_ == BanditType::LOOP) {
        // Matches your Python: LOOP returns next_layer_aux directly
        return next_layer_aux;
    }
    return get_gittins_denominator_aux_line(next_layer_aux);
}

MarkovChain::GittinsResult MarkovChain::get_stopping_time_and_gittins_parts(
    double state,
    double next_layer_numerator,
    double next_layer_aux) const
{
    MarkovChain start_chain = get_subchain(state, std::numeric_limits<double>::infinity());

    int best_time = 0;
    double best_num = 0.0;
    double best_den_aux = 0.0;
    double best_gi = -std::numeric_limits<double>::infinity();

    const auto& times = start_chain.all_times_;
    for (std::size_t i = 0; i < times.size(); ++i) {
        MarkovChain end_chain =
            start_chain.get_subchain(-std::numeric_limits<double>::infinity(), times[i]);

        const double numerator = end_chain.get_gittins_numerator(next_layer_numerator);
        const double den_aux = end_chain.get_gittins_denominator_aux(next_layer_aux);
        const double denominator = 1.0 / (1.0 - beta) - den_aux;
        const double gi = numerator / denominator;

        if (gi > best_gi) {
            best_gi = gi;
            best_time = times[i];
            best_num = numerator;
            best_den_aux = den_aux;
        }
    }

    return {best_time, best_num, best_den_aux};
}

// ======== Private helpers ========

std::vector<int> MarkovChain::compute_all_times(const std::vector<int>& done_times,
                                                  const std::vector<int>& fail_times)
{
    std::vector<int> all = done_times;
    all.insert(all.end(), fail_times.begin(), fail_times.end());
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    return all;
}

void MarkovChain::normalize_inputs() {
    auto has_zero = [](const std::vector<int>& v) {
        return std::find(v.begin(), v.end(), 0) != v.end();
    };

    // Ensure time 0 exists in both lists (like the Python code)
    if (!has_zero(done_times_)) {
        done_times_.insert(done_times_.begin(), 0);
        done_transitions_.insert(done_transitions_.begin(), 0.0);
    }
    if (!has_zero(fail_times_)) {
        fail_times_.insert(fail_times_.begin(), 0);
        fail_transitions_.insert(fail_transitions_.begin(), 0.0);
    }

    // Tail adjustment logic (mirrors Python)
    const int last_done_time = done_times_.empty() ? 0 : done_times_.back();
    const int last_fail_time = fail_times_.empty() ? 0 : fail_times_.back();

    if (last_fail_time == last_done_time) {
        if (!fail_transitions_.empty() && !done_transitions_.empty()) {
            fail_transitions_.back() = 1.0 - done_transitions_.back();
        }
    } else if (last_fail_time < last_done_time) {
        fail_times_.push_back(last_done_time);
        const double last_done_prob = done_transitions_.empty() ? 0.0 : done_transitions_.back();
        fail_transitions_.push_back(1.0 - last_done_prob);
    } else { // last_fail_time > last_done_time
        if (!fail_transitions_.empty()) {
            fail_transitions_.back() = 1.0;
        }
    }

    all_times_ = compute_all_times(done_times_, fail_times_);

    // Assertions equivalent to your Python asserts
    assert(done_transitions_.size() == done_times_.size());
    assert(fail_transitions_.size() == fail_times_.size());

    bool assert_passed = true;
    for (double p : done_transitions_) {
        if(p < 0.0 || p > 1.0) {
            assert_passed = false;
        }
    }
    for (double p : fail_transitions_) {
        if(p < 0.0 || p > 1.0) {
            assert_passed = false;
        }
    }
    
    if(!assert_passed) {
        cout << "MarkovChain assertion failed! Transitions out of range [0, 1]:" << std::endl;
        cout << "Done transitions: ";
        for (double p : done_transitions_) {
            cout << p << " ";
        }
        cout << std::endl;
        cout << "Fail transitions: ";
        for (double p : fail_transitions_) {
            cout << p << " ";
        }
        cout << std::endl;
    }
    assert(assert_passed);
}

double MarkovChain::get_gittins_numerator_loop(double next_layer_numerator) const {
    const std::size_t size = all_times_.size();
    arr M(size, size);
    arr rhs(size);
    M.setZero();
    rhs.setZero();

    std::size_t done_index = 0;
    std::size_t fail_index = 0;

    for (std::size_t i = 0; i < all_times_.size(); ++i) {
        const int t1 = all_times_[i];
        const int t2 = (i + 1 < all_times_.size()) ? all_times_[i + 1] : t1;

        double p_done = 0.0;
        double p_fail = 0.0;

        if (done_index < done_times_.size() && t1 == done_times_[done_index]) {
            p_done = done_transitions_[done_index];
            ++done_index;
        }
        if (fail_index < fail_times_.size() && t1 == fail_times_[fail_index]) {
            p_fail = fail_transitions_[fail_index];
            ++fail_index;
        }

        const double next_prob = 1.0 - p_done - p_fail;

        // matrix[i, 0] = beta * p_fail
        M(i, 0) = beta * p_fail;
        // rhs[i] = beta * p_done * next_layer_numerator
        rhs(i) = beta * p_done * next_layer_numerator;

        if (i + 1 < all_times_.size()) {
            // matrix[i, i+1] = beta^(t2-t1) * next_prob
            M(i, i + 1) = std::pow(beta, t2 - t1) * next_prob;
        }
    }

    // Build (I - M)
    arr IminusM(size, size);
    IminusM.setId();  // Identity matrix
    IminusM -= M;     // I - M

    arr a = inverse(IminusM) * rhs;
    return a(0);
}

double MarkovChain::get_gittins_numerator_line(double next_layer_numerator) const {
    const std::size_t size = all_times_.size();
    arr M(size, size);
    arr rhs(size);
    M.setZero();
    rhs.setZero();

    std::size_t done_index = 0;
    std::size_t fail_index = 0;

    for (std::size_t i = 0; i < all_times_.size(); ++i) {
        const int t1 = all_times_[i];
        const int t2 = (i + 1 < all_times_.size()) ? all_times_[i + 1] : t1;

        double p_done = 0.0;
        double p_fail = 0.0;

        if (done_index < done_times_.size() && t1 == done_times_[done_index]) {
            p_done = done_transitions_[done_index];
            ++done_index;
        }
        if (fail_index < fail_times_.size() && t1 == fail_times_[fail_index]) {
            p_fail = fail_transitions_[fail_index];
            ++fail_index;
        }

        const double next_prob = 1.0 - p_done - p_fail;

        // rhs[i] = beta * p_done * next_layer_numerator
        rhs(i) = beta * p_done * next_layer_numerator;

        if (i + 1 < all_times_.size()) {
            // matrix[i, i+1] = beta^(t2-t1) * next_prob
            M(i, i + 1) = std::pow(beta, t2 - t1) * next_prob;
        }
    }

    // Build (I - M)
    arr IminusM(size, size);
    IminusM.setId();  // Identity matrix
    IminusM -= M;     // I - M

    arr a = inverse(IminusM) * rhs;
    return a(0);
}

double MarkovChain::get_gittins_denominator_aux_line(double next_layer_aux) const {
    const std::size_t size = all_times_.size();
    arr M(size, size);
    arr rhs(size);
    M.setZero();
    rhs.setZero();

    std::size_t done_index = 0;
    std::size_t fail_index = 0;

    for (std::size_t i = 0; i < all_times_.size(); ++i) {
        const int t1 = all_times_[i];
        const int t2 = (i + 1 < all_times_.size()) ? all_times_[i + 1] : t1;

        double p_done = 0.0;
        double p_fail = 0.0;

        if (done_index < done_times_.size() && t1 == done_times_[done_index]) {
            p_done = done_transitions_[done_index];
            ++done_index;
        }
        if (fail_index < fail_times_.size() && t1 == fail_times_[fail_index]) {
            p_fail = fail_transitions_[fail_index];
            ++fail_index;
        }

        const double next_prob = 1.0 - p_done - p_fail;

        // rhs[i] = beta*p_done*next_layer_aux + beta*p_fail*(1/(1-beta))
        rhs(i) = beta * p_done * next_layer_aux
               + beta * p_fail * (1.0 / (1.0 - beta));

        if (i + 1 < all_times_.size()) {
            // matrix[i, i+1] = beta^(t2-t1) * next_prob
            M(i, i + 1) = std::pow(beta, t2 - t1) * next_prob;
        }
    }

    // Build (I - M)
    arr IminusM(size, size);
    IminusM.setId();  // Identity matrix
    IminusM -= M;     // I - M

    arr c = inverse(IminusM) * rhs;
    return c(0);
}
