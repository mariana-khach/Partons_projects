//
// Created by Mariana Khachatryan on 9/23/26.
//

#ifndef OBSERVABLE_RESULT_TORCH_H
#define OBSERVABLE_RESULT_TORCH_H

#include <partons/beans/List.h>
#include <partons/utils/type/PhysicalType.h>
#include <partons/utils/type/PhysicalUnit.h>
#include <torch/torch.h>

#include <string>

/**
 * @class ObservableResultTorch
 *
 * @brief Tensor counterpart of PARTONS' ObservableResult<KinematicType> — the
 *        result bean the differentiable chain returns.
 *
 * A SIBLING of ObservableResult, not an instantiation of it, for two reasons
 * that are worth stating because they are not obvious:
 *
 *  - ObservableResult's payload is `PhysicalType<double> m_value`. The value
 *    type is NOT a template parameter, so no instantiation of it can carry a
 *    torch::Tensor.
 *  - Its base Result<K> holds a SINGULAR `KinematicType m_kinematic` and needs
 *    operator< (for sortResultList()) and a const toString(); List<K> provides
 *    neither. A batched result is therefore a different shape, not a different
 *    instantiation.
 *
 * Same relationship ObservableTorch<K> already has to Observable<K,R>.
 *
 * Carries the whole batch: one [N] tensor and the N kinematics it was computed
 * at. The scalar chain instead returns List<ObservableResult>, one bean per
 * point, because Result<K>'s kinematic is singular and each point is computed
 * on its own worker thread. Ours is one object per call, one autograd graph,
 * no threads.
 *
 * Deliberately does NOT carry Result<K>'s channel type or result-info fields:
 * those exist so PARTONS can serialize results to its database and reports,
 * and nothing on the tensor path reads them. The computation module name is
 * kept because it costs one string and makes a result self-identifying in a
 * log.
 */
template<typename KinematicType>
class ObservableResultTorch {

public:

    ObservableResultTorch() :
            m_value(PARTONS::PhysicalUnit::UNDEFINED) {
    }

    ObservableResultTorch(const PARTONS::PhysicalType<torch::Tensor>& value,
            const PARTONS::List<KinematicType>& kinematics,
            const std::string& computationModuleName = std::string()) :
            m_value(value), m_kinematics(kinematics),
            m_computationModuleName(computationModuleName) {
    }

    /**
     * The observable value and its unit: an [N] tensor (or 0-d for the
     * single-kinematic entry point), grad-connected to the NN parameters.
     *
     * The unit is live information, not decoration — it is what lets the
     * cross-section leaves convert with makeSameUnitAs(PhysicalUnit::NB)
     * instead of a hand-copied constant, and what makes adding a GeV^-2 to an
     * nb throw instead of silently producing a wrong number.
     */
    const PARTONS::PhysicalType<torch::Tensor>& getValue() const {
        return m_value;
    }

    /**
     * Bare tensor, unit dropped — for callers that only want the number.
     *
     * Returned BY VALUE, and it must be: PhysicalType::getValue() itself
     * returns T by value, so handing back a reference here would bind to that
     * temporary and dangle. (It does not merely read wrong — it corrupts the
     * heap, "pointer being freed was not allocated", because a torch::Tensor
     * is a refcounted handle whose destructor then runs on a dead object.)
     * A Tensor copy is an intrusive_ptr bump, so by-value costs nothing and
     * shares the same storage and autograd graph.
     */
    torch::Tensor getTensor() const {
        return m_value.getValue();
    }

    PARTONS::PhysicalUnit::Type getUnit() const {
        return m_value.getUnit();
    }

    /** The N kinematics this batch was evaluated at, in tensor-row order. */
    const PARTONS::List<KinematicType>& getKinematics() const {
        return m_kinematics;
    }

    const std::string& getComputationModuleName() const {
        return m_computationModuleName;
    }

    void setComputationModuleName(const std::string& name) {
        m_computationModuleName = name;
    }

private:

    PARTONS::PhysicalType<torch::Tensor> m_value;
    PARTONS::List<KinematicType> m_kinematics;
    std::string m_computationModuleName;
};

#endif /* OBSERVABLE_RESULT_TORCH_H */