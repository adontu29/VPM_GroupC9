
These files were generated with the following setup:

### Properties of the Vortex Ring
```python
ring_center     = np.array([0.0, 0.0, 0.0])   # m, center of the vortex ring
ring_radius     = 1.0               # m, radius of the vortex ring
ring_strength   = 1.0               # m²/s, vortex strength
ring_thickness  = 0.2*ring_radius   # m, thickness of the vortex ring
```

### Particle Distribution Setup
```python
Re = 7500                                   # Reynolds number
particle_distance  = 0.25*ring_thickness    # m
particle_radius    = 0.8*particle_distance**0.5  # m
particle_viscosity = ring_strength/Re       # m²/s, kinematic viscosity
time_step_size     = 5 * particle_distance**2/ring_strength  # s
n_time_steps       = int( 20*ring_radius**2 / ring_strength / time_step_size)
```