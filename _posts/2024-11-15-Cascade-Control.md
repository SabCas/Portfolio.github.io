# Nonlinear Cascade Control: From Basic Controllers to Advanced Techniques in Autonomous Systems

Control systems play a crucial role in guiding the behavior of dynamic systems. In particular, nonlinear systems, which are common in robotics, aerospace, and autonomous vehicles, present unique challenges. Cascade control is a strategy that divides the control problem into multiple layers, improving both stability and performance. In this blog post, we will explore basic control strategies, including P, PI, PID, and the nonlinear cascade controller (as discussed in Schoellig's work), shedding light on how they are applied in autonomous systems like drones.

## 1. Basic Controllers: P, PI, and PID

Before diving into nonlinear cascade control, it's important to understand the building blocks of control systems—namely Proportional (P), Proportional-Integral (PI), and Proportional-Integral-Derivative (PID) controllers. These are often the first controllers that engineers use to stabilize systems, even in nonlinear environments.

### Proportional (P) Controller

A P-controller is the simplest form of a feedback controller. It applies a correction proportional to the error, which is the difference between the desired setpoint and the actual output. The control law for a P-controller is:

$$
u(t) = K_p e(t)
$$

where:

- \( u(t) \) is the control input,
- \( K_p \) is the proportional gain,
- \( e(t) = r(t) - y(t) \) is the error, where \( r(t) \) is the reference (desired output) and \( y(t) \) is the current output.

The P-controller provides a fast response but can lead to steady-state errors (offsets from the desired setpoint) if used alone. To address this, the PI and PID controllers are used.

### Proportional-Integral (PI) Controller

The PI-controller adds an integral term to the P-controller, which accumulates the error over time, effectively eliminating steady-state error. Its control law is:

$$
u(t) = K_p e(t) + K_i \int e(t) \, dt
$$

where:

- \( K_i \) is the integral gain,
- \( \int e(t) \, dt \) represents the accumulated error over time.

This allows the controller to eliminate long-term offsets but can introduce overshoot and oscillations if the integral gain is too large.

### Proportional-Integral-Derivative (PID) Controller

The PID-controller is the most widely used feedback controller. It combines the P, I, and D terms to balance the immediate response, error accumulation, and rate of change. The control law is:

$$
u(t) = K_p e(t) + K_i \int e(t) \, dt + K_d \frac{d e(t)}{dt}
$$

where:

- \( K_d \) is the derivative gain,
- \( \frac{d e(t)}{dt} \) is the rate of change of the error.

The D-term helps predict future errors, improving the system's stability and response time. However, if not tuned properly, it can amplify noise in the system.

## 2. Nonlinear Cascade Control

While PID controllers are effective for many linear systems, they become inadequate when dealing with nonlinear dynamics. Nonlinear systems exhibit complex behaviors, such as multiple equilibria and unpredictable responses, which cannot be fully captured by linear controllers like PID. To tackle this, nonlinear cascade control is often employed.

### Cascade Control Overview

Cascade control involves two or more loops: an outer loop (master controller) and one or more inner loops (slave controllers). The outer loop generates a reference for the inner loop, which regulates a specific part of the system. In nonlinear systems, this division helps simplify the problem and improves system response.

### Mathematical Model of Cascade Control

Consider a nonlinear system described by the following equations:

$$
\dot{x}_{\text{outer}} = f_{\text{outer}}(x_{\text{outer}}, u_{\text{outer}})
$$

$$
\dot{x}_{\text{inner}} = f_{\text{inner}}(x_{\text{inner}}, u_{\text{inner}})
$$

where:

- \( x_{\text{outer}} \) and \( x_{\text{inner}} \) are the states of the outer and inner loops,
- \( u_{\text{outer}} \) is the input to the outer loop,
- \( u_{\text{inner}} \) is the input to the inner loop.

The outer loop controller generates a reference input \( u_{\text{outer}} \) for the inner loop, which regulates the system according to:

$$
u_{\text{inner}} = k_{\text{inner}}(x_{\text{inner}}, u_{\text{outer}})
$$

In nonlinear systems, the controller gains \( k_{\text{outer}} \) and \( k_{\text{inner}} \) must be carefully designed to account for the system's dynamics.
