// BanditProcess.cpp
#include "BanditProcess.h"          // adjust path if needed
#include <limits>
#include <stdexcept>

namespace rai {

std::pair<int, double> BanditProcess::compute_gittins_index(double state) const {
    if (empty) {
        // You can decide what makes sense here; throwing is safest.
        throw std::runtime_error("BanditProcess::compute_gittins_index called on empty process");
        // return {1, -2.0};
    }

    // Matches Python:
    // next_layer_numerator = 1/(1-beta)
    // next_layer_aux = 0.0
    double next_layer_numerator = 1.0 / (1.0 - beta);
    double next_layer_aux = 0.0;

    int stopping_time = 0;
    double numerator = next_layer_numerator;
    double denominator_aux = next_layer_aux;

    const double neg_inf = -std::numeric_limits<double>::infinity();

    // Python: for i in reversed(range(len(self.markov_chains))):
    // i = last .. 0
    for (int i = static_cast<int>(markovChains.N) - 1; i >= 0; --i) {
        const MarkovChain& chain = markovChains(i);
        cout << "Computing Gittins index for chain " << i << " with state " << state << endl;
        chain.print_arrays();

        // Python:
        // chain_state = self.state if i == 0 else 0
        const double chain_state = (i == 0) ? state : 0.0;

        // stopping_time, numerator, denominator_aux = chain.get_stopping_time_and_gittins_parts(...)
        auto res = chain.get_stopping_time_and_gittins_parts(
            chain_state,
            next_layer_numerator,
            next_layer_aux
        );

        stopping_time = res.stopping_time;
        numerator = res.numerator;
        denominator_aux = res.denominator_aux;

        // Python special handling for LOOP:
        // if chain.type == LOOP:
        //     chain = chain.get_subchain(-inf, stopping_time)
        //     numerator = chain.get_gittins_numerator(next_layer_numerator)
        //     denominator_aux = chain.get_gittins_denominator_aux(next_layer_aux)
        //
        // NOTE: This assumes your MarkovChain exposes its type, either via
        // chain.type (public) or chain.get_type(). Adjust the line below accordingly.
        const bool isLoop =
            (chain.get_type() == BanditType::LOOP); // <-- change to chain.get_type()==... if needed

        if (isLoop) {
            MarkovChain truncated = chain.get_subchain(neg_inf, stopping_time);
            numerator = truncated.get_gittins_numerator(next_layer_numerator);
            denominator_aux = truncated.get_gittins_denominator_aux(next_layer_aux);
        }

        // Python:
        // next_layer_numerator = numerator
        // next_layer_aux = denominator_aux
        next_layer_numerator = numerator;
        next_layer_aux = denominator_aux;
    }

    // Python:
    // gittins_index = numerator/(1/(1-beta)-denominator_aux)
    const double denom = 1.0 / (1.0 - beta) - denominator_aux;
    const double gittins_index = numerator / denom;
    
    return std::make_pair(stopping_time, gittins_index);
}

} // namespace rai
