import pytest
import random as r
import numpy as np


def test_full_model():
    for x in range(100):
        # Get random variables
        a = r.randint(1, 100)
        k = r.randint(1, 100)
        y = r.randint(1, 100)
        W = r.randint(1, 100)
        j = r.randint(1, 100)
        z = r.randint(1, 100)
        f = r.randint(1, 100)
        b = r.randint(1, 100)
        A_e = r.randint(1, 100)
        A_p = r.randint(1, 100)
        S = r.randint(1, 100)
        g = r.randint(1, 100)
        S_e = r.randint(1, 100)

        # Define what we know
        E_u = A_p*y/(A_e*z)
        E_b = S*b*k*A_p*y/\
              (S*a*A_e*k*z + W*a*A_e*f*z + W*A_e*f*j*z)
        E = (W*a*b*A_p*y + W*b*j*A_p*y)/\
            (S*a*A_e*k*z + W*a*A_e*f*z + W*A_e*f*j*z)
        P_u = (S*a*k*y*z + W*a*b*f*y + W*a*f*y*z + W*b*f*j*y + W*f*j*y*z)\
              / (S*a*g*k*z + W*a*f*g*z + W*f*g*j*z)
        P_b = (S*a*k*y*z + W*a*b*f*y + W*a*f*y*z + W*b*f*j*y + W*f*j*y*z) /\
              (S*a**2*k*z + W*a**2*f*z + W*a*f*j*z)
        P = (S*W*a**2*k*y*z + S*W*a*j*k*y*z + W**2*a**2*b*f*y + W**2*a**2*f*y*z + 2*W**2*a*b*f*j*y + 2*W**2*a*f*j*y*z +
         W**2*b*f*j**2*y + W**2*f*j**2*y*z)/(S**2*a**2*k**2*z + S*W*a**2*S_e*k**2*z + S*W*a**2*f*k*z + S*W*a*f*j*k*z
                                             + W**2*a**2*S_e*f*k*z + W**2*a*S_e*f*j*k*z)

        # Test outputs
        assert np.allclose(y - k*S_e*P - (k/W)*S*P + j*P_b + f*(A_e/A_p)*E, 0), "Failed on dP/dt"
        assert np.allclose(k*S_e*P + (k/W)*S*P - j*P_b - a*P_b, 0), "Failed on dP_b/dt"
        assert np.allclose(a*P_b - g*P_u, 0), "Failed on dP_u/dt"
        var = g*(A_p/A_e)*P_u - b*E_u + a*E_b - z*E_u
        assert np.allclose(g*(A_p/A_e)*P_u - b*E_u + a*E_b - z*E_u, 0), f"Failed on dE/dt {var}"
        assert np.allclose(b*E_u - (k/W)*S*E + j*E_b - f*E, 0), "Failed on dE_b/dt"
        assert np.allclose((k/W)*S*E - j*E_b - a*E_b, 0), "Failed on dE_u/dt"


def test_four_model():
    for i in range(100):
        # Set up variables
        y = r.randint(1, 100)
        W = r.randint(1, 100)
        a = r.randint(1, 100)
        j = r.randint(1, 100)
        S_e = r.randint(1, 100)
        k = r.randint(1, 100)
        S = r.randint(1, 100)
        z = r.randint(1, 100)

        # Set up what we know
        P = (y*W*a + y*W*j) / (S_e*a*k*W + a*k*S)
        P_b = y/a
        P_u = y/z

        # Test
        dP_dt = y - k*S_e*P - (k/W)*S*P + j*P_b
        dPb_dt = k*S_e*P + (k/W)*S*P - j*P_b - a*P_b
        dPu_dt = a*P_b - z*P_u

        assert np.allclose(dP_dt, 0), "Failed on dP/dt"
        assert np.allclose(dPb_dt, 0), "Failed on dP_b/dt"
        assert np.allclose(dPu_dt, 0), "Failed on dP_u/dt"

