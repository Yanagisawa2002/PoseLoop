# PoseLoop M5-G0 mathematical conventions

## Pose convention

PoseLoop stores a model/object-to-camera transform

\[
T_{c\leftarrow o}=\begin{bmatrix}R_{c\leftarrow o}&t_{c\leftarrow o}\\0&1\end{bmatrix},
\qquad x_c=R_{c\leftarrow o}x_o+t_{c\leftarrow o}.
\]

Rotations act on column vectors. Nested JSON matrices are written as row-major
4 by 4 lists, translation is in the last column, and M5 uses metres internally.
The existing cross-camera convention is

\[
T_{w\leftarrow o}=T_{c\leftarrow w}^{-1}T_{c\leftarrow o},\qquad
T_{c_t\leftarrow o}=T_{c_t\leftarrow w}T_{w\leftarrow o}.
\]

## Right quotient by object symmetry

An object-frame symmetry acts on the right:

\[
T_{\mathrm{equivalent}}=T S.
\]

The finite groups are identity, C2 rotations of 0 and 180 degrees about object
`+z`, and C4 rotations of 0, 90, 180, and 270 degrees. Each element is an
explicit homogeneous 4 by 4 transform. For a predicted pose and measurement,
finite representative selection minimizes

\[
\left\|\log\left(T_{\mathrm{pred}}^{-1}T_{\mathrm{meas}}S\right)\right\|.
\]

For `CONTINUOUS_AXIAL`, the camera-frame direction
`R_c_o @ [0, 0, 1]` is observable and directed; `+z` and `-z` are not identified.
Any right rotation about object `+z` is unobservable. Rotation accuracy is the
angle between transformed `+z` axes. The measurement update removes the body
`z` rotational innovation and never learns body `z` angular velocity from the
arbitrary measurement gauge. A continuous output gauge is maintained only for
numerical continuity and visualization; it is not a recovered physical yaw.

The vendored BOP evaluator discretizes continuous symmetry. M5 therefore uses
the exact observable-axis residual for estimation and primary SO(2) quotient
accuracy. Synthetic normalized MSSD is reported only as a compatibility
diagnostic where finite symmetries make it applicable; it is not an official
BOP result.

## Constant body twist

The temporal state uses body twist `xi_b` and predicts causally with variable
time step:

\[
T_{k+1|k}=T_k\operatorname{Exp}(\xi_b\Delta t),\qquad
r_k=\operatorname{Log}(T_{k+1|k}^{-1}T_{\mathrm{meas},k}).
\]

SO(3) logarithms use SciPy's rotation-vector implementation and the SE(3)
Jacobian is evaluated with small-angle series where necessary. Tests cover
zero, near-zero, and near-pi rotations. No optimization failure may fall back
to ground truth.
