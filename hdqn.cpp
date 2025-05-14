#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <cmath>
#include <memory>
#include <string>
#include <stdexcept>

// c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3.12 -m pybind11 --includes) hdqn.cpp -o dqn_module$(python3.12-config --extension-suffix)

// --- Dominio ---

struct State {
    std::vector<double> features;
    explicit State(std::vector<double> features) : features(std::move(features)) {}
};

struct Action {
    int id;
    explicit Action(int id) : id(id) {}
};

class DQNModel {
public:
    DQNModel(size_t input_size, size_t output_size)
        : input_size_(input_size), output_size_(output_size), weights_(input_size * output_size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.1, 0.1);
        for (auto& w : weights_) w = dis(gen);
    }

    std::vector<double> predict(const State& state) const {
        if (state.features.size() != input_size_) {
            throw std::runtime_error("Tamaño de entrada inválido");
        }
        std::vector<double> q_values(output_size_, 0.0);
        for (size_t i = 0; i < output_size_; ++i) {
            for (size_t j = 0; j < input_size_; ++j) {
                q_values[i] += state.features[j] * weights_[i * input_size_ + j];
            }
            q_values[i] = std::tanh(q_values[i]);
        }
        return q_values;
    }

    void update(const State& state, const Action& action, double target, double learning_rate) {
        auto q_values = predict(state);
        double error = target - q_values[action.id];
        for (size_t j = 0; j < input_size_; ++j) {
            weights_[action.id * input_size_ + j] += learning_rate * error * state.features[j];
        }
    }

private:
    size_t input_size_;
    size_t output_size_;
    std::vector<double> weights_;
};

// --- Puertos ---

class ModelRepository {
public:
    virtual void save(const DQNModel& model) = 0;
    virtual std::optional<DQNModel> load() = 0;
    virtual ~ModelRepository() = default;
};

class DQNTrainer {
public:
    virtual void train(const std::vector<State>& states, const std::vector<Action>& actions,
                      const std::vector<double>& rewards, double gamma, double learning_rate) = 0;
    virtual ~DQNTrainer() = default;
};

class DQNPredictor {
public:
    virtual Action predict(const State& state) = 0;
    virtual ~DQNPredictor() = default;
};

// --- Adaptadores ---

class InMemoryModelRepository : public ModelRepository {
public:
    void save(const DQNModel& model) override {
        saved_model_ = model;
    }

    std::optional<DQNModel> load() override {
        return saved_model_;
    }

private:
    std::optional<DQNModel> saved_model_;
};

// --- Capa de aplicación ---

class DQNService : public DQNTrainer, public DQNPredictor {
public:
    // Constructor principal
    DQNService(std::unique_ptr<ModelRepository> repository, size_t input_size, size_t output_size)
        : repository_(std::move(repository)) {
        auto loaded_model = repository_->load();
        if (loaded_model) {
            model_ = std::make_unique<DQNModel>(*loaded_model);
        } else {
            model_ = std::make_unique<DQNModel>(input_size, output_size);
        }
    }

    // Constructor para pybind11: crea un InMemoryModelRepository por defecto
    DQNService(size_t input_size, size_t output_size)
        : DQNService(std::make_unique<InMemoryModelRepository>(), input_size, output_size) {}

    void train(const std::vector<State>& states, const std::vector<Action>& actions,
               const std::vector<double>& rewards, double gamma, double learning_rate) override {
        if (states.size() != actions.size() || states.size() != rewards.size()) {
            throw std::runtime_error("Tamaños de entrada inconsistentes");
        }

        for (size_t t = 0; t < states.size() - 1; ++t) {
            auto q_values = model_->predict(states[t]);
            double target = rewards[t];
            if (t < states.size() - 1) {
                auto next_q_values = model_->predict(states[t + 1]);
                target += gamma * *std::max_element(next_q_values.begin(), next_q_values.end());
            }
            model_->update(states[t], actions[t], target, learning_rate);
        }
        repository_->save(*model_);
    }

    Action predict(const State& state) override {
        auto q_values = model_->predict(state);
        int best_action = std::distance(q_values.begin(),
                                       std::max_element(q_values.begin(), q_values.end()));
        return Action(best_action);
    }

private:
    std::unique_ptr<ModelRepository> repository_;
    std::unique_ptr<DQNModel> model_;
};

// --- Entorno simulado ---

class SimpleEnvironment {
public:
    SimpleEnvironment(size_t state_size, size_t action_size)
        : state_size_(state_size), action_size_(action_size) {}

    std::tuple<State, double, bool> step(const Action& action) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        std::vector<double> new_features(state_size_);
        for (auto& f : new_features) f = dis(gen);
        State new_state(new_features);

        double reward = (action.id == optimal_action()) ? 1.0 : -0.1;
        bool done = dis(gen) < 0.1;

        return {new_state, reward, done};
    }

    State reset() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::vector<double> features(state_size_);
        for (auto& f : features) f = dis(gen);
        return State(features);
    }

private:
    size_t state_size_;
    size_t action_size_;

    int optimal_action() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, action_size_ - 1);
        return dis(gen);
    }
};

// --- Pybind11 ---

namespace py = pybind11;

PYBIND11_MODULE(dqn_module, m) {
    py::class_<State>(m, "State")
        .def(py::init<std::vector<double>>())
        .def_readwrite("features", &State::features);

    py::class_<Action>(m, "Action")
        .def(py::init<int>())
        .def_readwrite("id", &Action::id);

    py::class_<DQNService>(m, "DQNService")
        .def(py::init<size_t, size_t>())
        .def("train", &DQNService::train)
        .def("predict", &DQNService::predict);

    py::class_<SimpleEnvironment>(m, "SimpleEnvironment")
        .def(py::init<size_t, size_t>())
        .def("step", &SimpleEnvironment::step)
        .def("reset", &SimpleEnvironment::reset);
}