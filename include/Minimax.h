/**
 * minimax
 * Copyright (C) 2017  Joey Andres<yeojserdna@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <vector>
#include <numeric>
#include <utility>  // std::pair
#include <iterator>  // std::advance
#include <limits>  // -std::numeric_limits<double>::infinity()
#include <cmath>  // std::min, std::max

using std::vector;
using std::pair;

namespace minimax {

// TODO(jandres): be more generalized, e.g. more than two users.
/*!@class Minimax
 * @brief Minimax class.
 * @tparam S State data type.
 * @tparam A Action data type.
 */
template <class S, class A>
class Minimax {
 public:
  // Must be overridden.

  /**
   * Acquires set of action given state.
   * @param s State
   * @return Set of action available for the given state.
   */
  virtual set<A> getAction(const S& s) const = 0;

  /**
   * Given a state returns its value.
   * @param s State
   * @return Value of the given state. Probably terminal.
   */
  virtual double utility(const S& s) const = 0;

  /**
   * A way to determine if given state is terminal state.
   * @param s State
   * @return true if terminal state.
   */
  virtual bool isTerminal(const S& s) const = 0;

  /**
   * Given a state and action, retrieves the next state.
   * @param s State
   * @param a Action
   * @return New state.
   */
  virtual S result(const S& s, const A& a) const = 0;

 public:
  /**
   * Given a state, retrieves the best action and it's associated value.
   * @param s State
   * @return Action and value of chosen action.
   */
  pair<A, double> decision(const S& s) const;

 protected:
  /**
   * Value of s, minimizing the action chosen by "max" user.
   * @param s State
   * @return value.
   */
  double _getMinValue(const S& s) const;

  /**
   * Value of s, maximizing the action chosen by "min" user.
   * @param s State
   * @return value.
   */
  double _getMaxValue(const S& s) const;
};

template <class S, class A>
pair<A, double> Minimax<S, A>::decision(const S &s) const {
  auto actions = getAction(s);
  A maxAction = actions.begin();
  double maxValue = _getMinValue(result(s, maxAction));

  // TODO: Do std::advance(actions.begin(), 1); once we have more tests.
  for (auto& a : actions) {
    double currentValue = _getMinValue(result(s, a));
    if (currentValue >= maxValue) {
      maxAction = a;
      maxValue = currentValue;
    }
  }

  return pair<A, double>(maxAction, maxValue);
}

template <class S, class A>
double Minimax<S, A>::_getMinValue(const S& s) const {
  if (isTerminal(s)) {
    return utility(s);
  }

  double value = std::numeric_limits<double>::infinity();
  for (auto& a : getAction(s)) {
    value = std::min(value, _getMaxValue(result(s, a)));
  }

  return value;
}

template <class S, class A>
double Minimax<S, A>::_getMaxValue(const S& s) const {
  if (isTerminal(s)) {
    return utility(s);
  }

  double value = -std::numeric_limits<double>::infinity();
  for (auto& a : getAction(s)) {
    value = std::max(value, _getMinValue(result(s, a)));
  }

  return value;
}

}  // namespace minimax