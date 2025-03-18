import math

class VortexRingInstance:
    def __init__(self, X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t):
        self.x = X
        self.y = Y
        self.z = Z
        self.u = U
        self.v = V
        self.w = W
        self.wx = Wx
        self.wy = Wy 
        self.wz = Wz
        self.radius = Radius
        self.group_ID = Group_ID
        self.viscosity = Viscosity
        self.viscosity_t = Viscosity_t