��1
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype�
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%��8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu6
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
H
ShardedFilename
basename	
shard

num_shards
filename
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�
9
VarIsInitializedOp
resource
is_initialized
�"serve*1.14.02unknown8��,
~
input_1Placeholder*&
shape:�����������*
dtype0*1
_output_shapes
:�����������
�
Conv1_pad/Pad/paddingsConst*9
value0B."                               *
dtype0*
_output_shapes

:
q
Conv1_pad/PadPadinput_1Conv1_pad/Pad/paddings*
T0*1
_output_shapes
:�����������
�
-Conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"             *
_class
loc:@Conv1/kernel*
dtype0*
_output_shapes
:
�
+Conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *OS�*
_class
loc:@Conv1/kernel*
dtype0*
_output_shapes
: 
�
+Conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *OS>*
_class
loc:@Conv1/kernel*
dtype0*
_output_shapes
: 
�
5Conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform-Conv1/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@Conv1/kernel*
dtype0*&
_output_shapes
: 
�
+Conv1/kernel/Initializer/random_uniform/subSub+Conv1/kernel/Initializer/random_uniform/max+Conv1/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@Conv1/kernel*
_output_shapes
: 
�
+Conv1/kernel/Initializer/random_uniform/mulMul5Conv1/kernel/Initializer/random_uniform/RandomUniform+Conv1/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@Conv1/kernel*&
_output_shapes
: 
�
'Conv1/kernel/Initializer/random_uniformAdd+Conv1/kernel/Initializer/random_uniform/mul+Conv1/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@Conv1/kernel*&
_output_shapes
: 
�
Conv1/kernelVarHandleOp*
shape: *
shared_nameConv1/kernel*
_class
loc:@Conv1/kernel*
dtype0*
_output_shapes
: 
i
-Conv1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpConv1/kernel*
_output_shapes
: 
�
Conv1/kernel/AssignAssignVariableOpConv1/kernel'Conv1/kernel/Initializer/random_uniform*
_class
loc:@Conv1/kernel*
dtype0
�
 Conv1/kernel/Read/ReadVariableOpReadVariableOpConv1/kernel*
_class
loc:@Conv1/kernel*
dtype0*&
_output_shapes
: 
d
Conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
p
Conv1/Conv2D/ReadVariableOpReadVariableOpConv1/kernel*
dtype0*&
_output_shapes
: 
�
Conv1/Conv2DConv2DConv1_pad/PadConv1/Conv2D/ReadVariableOp*
paddingVALID*
T0*
strides
*/
_output_shapes
:���������pp 
�
bn_Conv1/gamma/Initializer/onesConst*
valueB *  �?*!
_class
loc:@bn_Conv1/gamma*
dtype0*
_output_shapes
: 
�
bn_Conv1/gammaVarHandleOp*
shape: *
shared_namebn_Conv1/gamma*!
_class
loc:@bn_Conv1/gamma*
dtype0*
_output_shapes
: 
m
/bn_Conv1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbn_Conv1/gamma*
_output_shapes
: 
�
bn_Conv1/gamma/AssignAssignVariableOpbn_Conv1/gammabn_Conv1/gamma/Initializer/ones*!
_class
loc:@bn_Conv1/gamma*
dtype0
�
"bn_Conv1/gamma/Read/ReadVariableOpReadVariableOpbn_Conv1/gamma*!
_class
loc:@bn_Conv1/gamma*
dtype0*
_output_shapes
: 
�
bn_Conv1/beta/Initializer/zerosConst*
valueB *    * 
_class
loc:@bn_Conv1/beta*
dtype0*
_output_shapes
: 
�
bn_Conv1/betaVarHandleOp*
shape: *
shared_namebn_Conv1/beta* 
_class
loc:@bn_Conv1/beta*
dtype0*
_output_shapes
: 
k
.bn_Conv1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbn_Conv1/beta*
_output_shapes
: 
�
bn_Conv1/beta/AssignAssignVariableOpbn_Conv1/betabn_Conv1/beta/Initializer/zeros* 
_class
loc:@bn_Conv1/beta*
dtype0
�
!bn_Conv1/beta/Read/ReadVariableOpReadVariableOpbn_Conv1/beta* 
_class
loc:@bn_Conv1/beta*
dtype0*
_output_shapes
: 
�
&bn_Conv1/moving_mean/Initializer/zerosConst*
valueB *    *'
_class
loc:@bn_Conv1/moving_mean*
dtype0*
_output_shapes
: 
�
bn_Conv1/moving_meanVarHandleOp*
shape: *%
shared_namebn_Conv1/moving_mean*'
_class
loc:@bn_Conv1/moving_mean*
dtype0*
_output_shapes
: 
y
5bn_Conv1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpbn_Conv1/moving_mean*
_output_shapes
: 
�
bn_Conv1/moving_mean/AssignAssignVariableOpbn_Conv1/moving_mean&bn_Conv1/moving_mean/Initializer/zeros*'
_class
loc:@bn_Conv1/moving_mean*
dtype0
�
(bn_Conv1/moving_mean/Read/ReadVariableOpReadVariableOpbn_Conv1/moving_mean*'
_class
loc:@bn_Conv1/moving_mean*
dtype0*
_output_shapes
: 
�
)bn_Conv1/moving_variance/Initializer/onesConst*
valueB *  �?*+
_class!
loc:@bn_Conv1/moving_variance*
dtype0*
_output_shapes
: 
�
bn_Conv1/moving_varianceVarHandleOp*
shape: *)
shared_namebn_Conv1/moving_variance*+
_class!
loc:@bn_Conv1/moving_variance*
dtype0*
_output_shapes
: 
�
9bn_Conv1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOpbn_Conv1/moving_variance*
_output_shapes
: 
�
bn_Conv1/moving_variance/AssignAssignVariableOpbn_Conv1/moving_variance)bn_Conv1/moving_variance/Initializer/ones*+
_class!
loc:@bn_Conv1/moving_variance*
dtype0
�
,bn_Conv1/moving_variance/Read/ReadVariableOpReadVariableOpbn_Conv1/moving_variance*+
_class!
loc:@bn_Conv1/moving_variance*
dtype0*
_output_shapes
: 
b
bn_Conv1/ReadVariableOpReadVariableOpbn_Conv1/gamma*
dtype0*
_output_shapes
: 
c
bn_Conv1/ReadVariableOp_1ReadVariableOpbn_Conv1/beta*
dtype0*
_output_shapes
: 
w
&bn_Conv1/FusedBatchNorm/ReadVariableOpReadVariableOpbn_Conv1/moving_mean*
dtype0*
_output_shapes
: 
}
(bn_Conv1/FusedBatchNorm/ReadVariableOp_1ReadVariableOpbn_Conv1/moving_variance*
dtype0*
_output_shapes
: 
�
bn_Conv1/FusedBatchNormFusedBatchNormConv1/Conv2Dbn_Conv1/ReadVariableOpbn_Conv1/ReadVariableOp_1&bn_Conv1/FusedBatchNorm/ReadVariableOp(bn_Conv1/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:���������pp : : : : *
is_training( 
S
bn_Conv1/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
l
Conv1_relu/Relu6Relu6bn_Conv1/FusedBatchNorm*
T0*/
_output_shapes
:���������pp 
�
Iexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"             *;
_class1
/-loc:@expanded_conv_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Gexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *���*;
_class1
/-loc:@expanded_conv_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Gexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *��>*;
_class1
/-loc:@expanded_conv_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Qexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformIexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*;
_class1
/-loc:@expanded_conv_depthwise/depthwise_kernel*
dtype0*&
_output_shapes
: 
�
Gexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform/subSubGexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform/maxGexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*;
_class1
/-loc:@expanded_conv_depthwise/depthwise_kernel*
_output_shapes
: 
�
Gexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulQexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformGexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*;
_class1
/-loc:@expanded_conv_depthwise/depthwise_kernel*&
_output_shapes
: 
�
Cexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniformAddGexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform/mulGexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*;
_class1
/-loc:@expanded_conv_depthwise/depthwise_kernel*&
_output_shapes
: 
�
(expanded_conv_depthwise/depthwise_kernelVarHandleOp*
shape: *9
shared_name*(expanded_conv_depthwise/depthwise_kernel*;
_class1
/-loc:@expanded_conv_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Iexpanded_conv_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp(expanded_conv_depthwise/depthwise_kernel*
_output_shapes
: 
�
/expanded_conv_depthwise/depthwise_kernel/AssignAssignVariableOp(expanded_conv_depthwise/depthwise_kernelCexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform*;
_class1
/-loc:@expanded_conv_depthwise/depthwise_kernel*
dtype0
�
<expanded_conv_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp(expanded_conv_depthwise/depthwise_kernel*;
_class1
/-loc:@expanded_conv_depthwise/depthwise_kernel*
dtype0*&
_output_shapes
: 
�
0expanded_conv_depthwise/depthwise/ReadVariableOpReadVariableOp(expanded_conv_depthwise/depthwise_kernel*
dtype0*&
_output_shapes
: 
�
'expanded_conv_depthwise/depthwise/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
�
/expanded_conv_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
!expanded_conv_depthwise/depthwiseDepthwiseConv2dNativeConv1_relu/Relu60expanded_conv_depthwise/depthwise/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:���������pp 
�
1expanded_conv_depthwise_BN/gamma/Initializer/onesConst*
valueB *  �?*3
_class)
'%loc:@expanded_conv_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
 expanded_conv_depthwise_BN/gammaVarHandleOp*
shape: *1
shared_name" expanded_conv_depthwise_BN/gamma*3
_class)
'%loc:@expanded_conv_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
Aexpanded_conv_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp expanded_conv_depthwise_BN/gamma*
_output_shapes
: 
�
'expanded_conv_depthwise_BN/gamma/AssignAssignVariableOp expanded_conv_depthwise_BN/gamma1expanded_conv_depthwise_BN/gamma/Initializer/ones*3
_class)
'%loc:@expanded_conv_depthwise_BN/gamma*
dtype0
�
4expanded_conv_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOp expanded_conv_depthwise_BN/gamma*3
_class)
'%loc:@expanded_conv_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
1expanded_conv_depthwise_BN/beta/Initializer/zerosConst*
valueB *    *2
_class(
&$loc:@expanded_conv_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
expanded_conv_depthwise_BN/betaVarHandleOp*
shape: *0
shared_name!expanded_conv_depthwise_BN/beta*2
_class(
&$loc:@expanded_conv_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
@expanded_conv_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpexpanded_conv_depthwise_BN/beta*
_output_shapes
: 
�
&expanded_conv_depthwise_BN/beta/AssignAssignVariableOpexpanded_conv_depthwise_BN/beta1expanded_conv_depthwise_BN/beta/Initializer/zeros*2
_class(
&$loc:@expanded_conv_depthwise_BN/beta*
dtype0
�
3expanded_conv_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpexpanded_conv_depthwise_BN/beta*2
_class(
&$loc:@expanded_conv_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
8expanded_conv_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB *    *9
_class/
-+loc:@expanded_conv_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
&expanded_conv_depthwise_BN/moving_meanVarHandleOp*
shape: *7
shared_name(&expanded_conv_depthwise_BN/moving_mean*9
_class/
-+loc:@expanded_conv_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Gexpanded_conv_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp&expanded_conv_depthwise_BN/moving_mean*
_output_shapes
: 
�
-expanded_conv_depthwise_BN/moving_mean/AssignAssignVariableOp&expanded_conv_depthwise_BN/moving_mean8expanded_conv_depthwise_BN/moving_mean/Initializer/zeros*9
_class/
-+loc:@expanded_conv_depthwise_BN/moving_mean*
dtype0
�
:expanded_conv_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp&expanded_conv_depthwise_BN/moving_mean*9
_class/
-+loc:@expanded_conv_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
;expanded_conv_depthwise_BN/moving_variance/Initializer/onesConst*
valueB *  �?*=
_class3
1/loc:@expanded_conv_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
*expanded_conv_depthwise_BN/moving_varianceVarHandleOp*
shape: *;
shared_name,*expanded_conv_depthwise_BN/moving_variance*=
_class3
1/loc:@expanded_conv_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Kexpanded_conv_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp*expanded_conv_depthwise_BN/moving_variance*
_output_shapes
: 
�
1expanded_conv_depthwise_BN/moving_variance/AssignAssignVariableOp*expanded_conv_depthwise_BN/moving_variance;expanded_conv_depthwise_BN/moving_variance/Initializer/ones*=
_class3
1/loc:@expanded_conv_depthwise_BN/moving_variance*
dtype0
�
>expanded_conv_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp*expanded_conv_depthwise_BN/moving_variance*=
_class3
1/loc:@expanded_conv_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
)expanded_conv_depthwise_BN/ReadVariableOpReadVariableOp expanded_conv_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
+expanded_conv_depthwise_BN/ReadVariableOp_1ReadVariableOpexpanded_conv_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
8expanded_conv_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp&expanded_conv_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
:expanded_conv_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp*expanded_conv_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
)expanded_conv_depthwise_BN/FusedBatchNormFusedBatchNorm!expanded_conv_depthwise/depthwise)expanded_conv_depthwise_BN/ReadVariableOp+expanded_conv_depthwise_BN/ReadVariableOp_18expanded_conv_depthwise_BN/FusedBatchNorm/ReadVariableOp:expanded_conv_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:���������pp : : : : *
is_training( 
e
 expanded_conv_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
"expanded_conv_depthwise_relu/Relu6Relu6)expanded_conv_depthwise_BN/FusedBatchNorm*
T0*/
_output_shapes
:���������pp 
�
=expanded_conv_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"             */
_class%
#!loc:@expanded_conv_project/kernel*
dtype0*
_output_shapes
:
�
;expanded_conv_project/kernel/Initializer/random_uniform/minConst*
valueB
 *���*/
_class%
#!loc:@expanded_conv_project/kernel*
dtype0*
_output_shapes
: 
�
;expanded_conv_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *��>*/
_class%
#!loc:@expanded_conv_project/kernel*
dtype0*
_output_shapes
: 
�
Eexpanded_conv_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform=expanded_conv_project/kernel/Initializer/random_uniform/shape*
T0*/
_class%
#!loc:@expanded_conv_project/kernel*
dtype0*&
_output_shapes
: 
�
;expanded_conv_project/kernel/Initializer/random_uniform/subSub;expanded_conv_project/kernel/Initializer/random_uniform/max;expanded_conv_project/kernel/Initializer/random_uniform/min*
T0*/
_class%
#!loc:@expanded_conv_project/kernel*
_output_shapes
: 
�
;expanded_conv_project/kernel/Initializer/random_uniform/mulMulEexpanded_conv_project/kernel/Initializer/random_uniform/RandomUniform;expanded_conv_project/kernel/Initializer/random_uniform/sub*
T0*/
_class%
#!loc:@expanded_conv_project/kernel*&
_output_shapes
: 
�
7expanded_conv_project/kernel/Initializer/random_uniformAdd;expanded_conv_project/kernel/Initializer/random_uniform/mul;expanded_conv_project/kernel/Initializer/random_uniform/min*
T0*/
_class%
#!loc:@expanded_conv_project/kernel*&
_output_shapes
: 
�
expanded_conv_project/kernelVarHandleOp*
shape: *-
shared_nameexpanded_conv_project/kernel*/
_class%
#!loc:@expanded_conv_project/kernel*
dtype0*
_output_shapes
: 
�
=expanded_conv_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpexpanded_conv_project/kernel*
_output_shapes
: 
�
#expanded_conv_project/kernel/AssignAssignVariableOpexpanded_conv_project/kernel7expanded_conv_project/kernel/Initializer/random_uniform*/
_class%
#!loc:@expanded_conv_project/kernel*
dtype0
�
0expanded_conv_project/kernel/Read/ReadVariableOpReadVariableOpexpanded_conv_project/kernel*/
_class%
#!loc:@expanded_conv_project/kernel*
dtype0*&
_output_shapes
: 
t
#expanded_conv_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
+expanded_conv_project/Conv2D/ReadVariableOpReadVariableOpexpanded_conv_project/kernel*
dtype0*&
_output_shapes
: 
�
expanded_conv_project/Conv2DConv2D"expanded_conv_depthwise_relu/Relu6+expanded_conv_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:���������pp
�
/expanded_conv_project_BN/gamma/Initializer/onesConst*
valueB*  �?*1
_class'
%#loc:@expanded_conv_project_BN/gamma*
dtype0*
_output_shapes
:
�
expanded_conv_project_BN/gammaVarHandleOp*
shape:*/
shared_name expanded_conv_project_BN/gamma*1
_class'
%#loc:@expanded_conv_project_BN/gamma*
dtype0*
_output_shapes
: 
�
?expanded_conv_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpexpanded_conv_project_BN/gamma*
_output_shapes
: 
�
%expanded_conv_project_BN/gamma/AssignAssignVariableOpexpanded_conv_project_BN/gamma/expanded_conv_project_BN/gamma/Initializer/ones*1
_class'
%#loc:@expanded_conv_project_BN/gamma*
dtype0
�
2expanded_conv_project_BN/gamma/Read/ReadVariableOpReadVariableOpexpanded_conv_project_BN/gamma*1
_class'
%#loc:@expanded_conv_project_BN/gamma*
dtype0*
_output_shapes
:
�
/expanded_conv_project_BN/beta/Initializer/zerosConst*
valueB*    *0
_class&
$"loc:@expanded_conv_project_BN/beta*
dtype0*
_output_shapes
:
�
expanded_conv_project_BN/betaVarHandleOp*
shape:*.
shared_nameexpanded_conv_project_BN/beta*0
_class&
$"loc:@expanded_conv_project_BN/beta*
dtype0*
_output_shapes
: 
�
>expanded_conv_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpexpanded_conv_project_BN/beta*
_output_shapes
: 
�
$expanded_conv_project_BN/beta/AssignAssignVariableOpexpanded_conv_project_BN/beta/expanded_conv_project_BN/beta/Initializer/zeros*0
_class&
$"loc:@expanded_conv_project_BN/beta*
dtype0
�
1expanded_conv_project_BN/beta/Read/ReadVariableOpReadVariableOpexpanded_conv_project_BN/beta*0
_class&
$"loc:@expanded_conv_project_BN/beta*
dtype0*
_output_shapes
:
�
6expanded_conv_project_BN/moving_mean/Initializer/zerosConst*
valueB*    *7
_class-
+)loc:@expanded_conv_project_BN/moving_mean*
dtype0*
_output_shapes
:
�
$expanded_conv_project_BN/moving_meanVarHandleOp*
shape:*5
shared_name&$expanded_conv_project_BN/moving_mean*7
_class-
+)loc:@expanded_conv_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Eexpanded_conv_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp$expanded_conv_project_BN/moving_mean*
_output_shapes
: 
�
+expanded_conv_project_BN/moving_mean/AssignAssignVariableOp$expanded_conv_project_BN/moving_mean6expanded_conv_project_BN/moving_mean/Initializer/zeros*7
_class-
+)loc:@expanded_conv_project_BN/moving_mean*
dtype0
�
8expanded_conv_project_BN/moving_mean/Read/ReadVariableOpReadVariableOp$expanded_conv_project_BN/moving_mean*7
_class-
+)loc:@expanded_conv_project_BN/moving_mean*
dtype0*
_output_shapes
:
�
9expanded_conv_project_BN/moving_variance/Initializer/onesConst*
valueB*  �?*;
_class1
/-loc:@expanded_conv_project_BN/moving_variance*
dtype0*
_output_shapes
:
�
(expanded_conv_project_BN/moving_varianceVarHandleOp*
shape:*9
shared_name*(expanded_conv_project_BN/moving_variance*;
_class1
/-loc:@expanded_conv_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Iexpanded_conv_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp(expanded_conv_project_BN/moving_variance*
_output_shapes
: 
�
/expanded_conv_project_BN/moving_variance/AssignAssignVariableOp(expanded_conv_project_BN/moving_variance9expanded_conv_project_BN/moving_variance/Initializer/ones*;
_class1
/-loc:@expanded_conv_project_BN/moving_variance*
dtype0
�
<expanded_conv_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp(expanded_conv_project_BN/moving_variance*;
_class1
/-loc:@expanded_conv_project_BN/moving_variance*
dtype0*
_output_shapes
:
�
'expanded_conv_project_BN/ReadVariableOpReadVariableOpexpanded_conv_project_BN/gamma*
dtype0*
_output_shapes
:
�
)expanded_conv_project_BN/ReadVariableOp_1ReadVariableOpexpanded_conv_project_BN/beta*
dtype0*
_output_shapes
:
�
6expanded_conv_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOp$expanded_conv_project_BN/moving_mean*
dtype0*
_output_shapes
:
�
8expanded_conv_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp(expanded_conv_project_BN/moving_variance*
dtype0*
_output_shapes
:
�
'expanded_conv_project_BN/FusedBatchNormFusedBatchNormexpanded_conv_project/Conv2D'expanded_conv_project_BN/ReadVariableOp)expanded_conv_project_BN/ReadVariableOp_16expanded_conv_project_BN/FusedBatchNorm/ReadVariableOp8expanded_conv_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:���������pp::::*
is_training( 
c
expanded_conv_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
6block_1_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"         `   *(
_class
loc:@block_1_expand/kernel*
dtype0*
_output_shapes
:
�
4block_1_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *�m�*(
_class
loc:@block_1_expand/kernel*
dtype0*
_output_shapes
: 
�
4block_1_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *�m>*(
_class
loc:@block_1_expand/kernel*
dtype0*
_output_shapes
: 
�
>block_1_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform6block_1_expand/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@block_1_expand/kernel*
dtype0*&
_output_shapes
:`
�
4block_1_expand/kernel/Initializer/random_uniform/subSub4block_1_expand/kernel/Initializer/random_uniform/max4block_1_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_1_expand/kernel*
_output_shapes
: 
�
4block_1_expand/kernel/Initializer/random_uniform/mulMul>block_1_expand/kernel/Initializer/random_uniform/RandomUniform4block_1_expand/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@block_1_expand/kernel*&
_output_shapes
:`
�
0block_1_expand/kernel/Initializer/random_uniformAdd4block_1_expand/kernel/Initializer/random_uniform/mul4block_1_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_1_expand/kernel*&
_output_shapes
:`
�
block_1_expand/kernelVarHandleOp*
shape:`*&
shared_nameblock_1_expand/kernel*(
_class
loc:@block_1_expand/kernel*
dtype0*
_output_shapes
: 
{
6block_1_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_1_expand/kernel*
_output_shapes
: 
�
block_1_expand/kernel/AssignAssignVariableOpblock_1_expand/kernel0block_1_expand/kernel/Initializer/random_uniform*(
_class
loc:@block_1_expand/kernel*
dtype0
�
)block_1_expand/kernel/Read/ReadVariableOpReadVariableOpblock_1_expand/kernel*(
_class
loc:@block_1_expand/kernel*
dtype0*&
_output_shapes
:`
m
block_1_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
$block_1_expand/Conv2D/ReadVariableOpReadVariableOpblock_1_expand/kernel*
dtype0*&
_output_shapes
:`
�
block_1_expand/Conv2DConv2D'expanded_conv_project_BN/FusedBatchNorm$block_1_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:���������pp`
�
(block_1_expand_BN/gamma/Initializer/onesConst*
valueB`*  �?**
_class 
loc:@block_1_expand_BN/gamma*
dtype0*
_output_shapes
:`
�
block_1_expand_BN/gammaVarHandleOp*
shape:`*(
shared_nameblock_1_expand_BN/gamma**
_class 
loc:@block_1_expand_BN/gamma*
dtype0*
_output_shapes
: 

8block_1_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_1_expand_BN/gamma*
_output_shapes
: 
�
block_1_expand_BN/gamma/AssignAssignVariableOpblock_1_expand_BN/gamma(block_1_expand_BN/gamma/Initializer/ones**
_class 
loc:@block_1_expand_BN/gamma*
dtype0
�
+block_1_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_1_expand_BN/gamma**
_class 
loc:@block_1_expand_BN/gamma*
dtype0*
_output_shapes
:`
�
(block_1_expand_BN/beta/Initializer/zerosConst*
valueB`*    *)
_class
loc:@block_1_expand_BN/beta*
dtype0*
_output_shapes
:`
�
block_1_expand_BN/betaVarHandleOp*
shape:`*'
shared_nameblock_1_expand_BN/beta*)
_class
loc:@block_1_expand_BN/beta*
dtype0*
_output_shapes
: 
}
7block_1_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_1_expand_BN/beta*
_output_shapes
: 
�
block_1_expand_BN/beta/AssignAssignVariableOpblock_1_expand_BN/beta(block_1_expand_BN/beta/Initializer/zeros*)
_class
loc:@block_1_expand_BN/beta*
dtype0
�
*block_1_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_1_expand_BN/beta*)
_class
loc:@block_1_expand_BN/beta*
dtype0*
_output_shapes
:`
�
/block_1_expand_BN/moving_mean/Initializer/zerosConst*
valueB`*    *0
_class&
$"loc:@block_1_expand_BN/moving_mean*
dtype0*
_output_shapes
:`
�
block_1_expand_BN/moving_meanVarHandleOp*
shape:`*.
shared_nameblock_1_expand_BN/moving_mean*0
_class&
$"loc:@block_1_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
>block_1_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_1_expand_BN/moving_mean*
_output_shapes
: 
�
$block_1_expand_BN/moving_mean/AssignAssignVariableOpblock_1_expand_BN/moving_mean/block_1_expand_BN/moving_mean/Initializer/zeros*0
_class&
$"loc:@block_1_expand_BN/moving_mean*
dtype0
�
1block_1_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_1_expand_BN/moving_mean*0
_class&
$"loc:@block_1_expand_BN/moving_mean*
dtype0*
_output_shapes
:`
�
2block_1_expand_BN/moving_variance/Initializer/onesConst*
valueB`*  �?*4
_class*
(&loc:@block_1_expand_BN/moving_variance*
dtype0*
_output_shapes
:`
�
!block_1_expand_BN/moving_varianceVarHandleOp*
shape:`*2
shared_name#!block_1_expand_BN/moving_variance*4
_class*
(&loc:@block_1_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Bblock_1_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_1_expand_BN/moving_variance*
_output_shapes
: 
�
(block_1_expand_BN/moving_variance/AssignAssignVariableOp!block_1_expand_BN/moving_variance2block_1_expand_BN/moving_variance/Initializer/ones*4
_class*
(&loc:@block_1_expand_BN/moving_variance*
dtype0
�
5block_1_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp!block_1_expand_BN/moving_variance*4
_class*
(&loc:@block_1_expand_BN/moving_variance*
dtype0*
_output_shapes
:`
t
 block_1_expand_BN/ReadVariableOpReadVariableOpblock_1_expand_BN/gamma*
dtype0*
_output_shapes
:`
u
"block_1_expand_BN/ReadVariableOp_1ReadVariableOpblock_1_expand_BN/beta*
dtype0*
_output_shapes
:`
�
/block_1_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_1_expand_BN/moving_mean*
dtype0*
_output_shapes
:`
�
1block_1_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp!block_1_expand_BN/moving_variance*
dtype0*
_output_shapes
:`
�
 block_1_expand_BN/FusedBatchNormFusedBatchNormblock_1_expand/Conv2D block_1_expand_BN/ReadVariableOp"block_1_expand_BN/ReadVariableOp_1/block_1_expand_BN/FusedBatchNorm/ReadVariableOp1block_1_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:���������pp`:`:`:`:`*
is_training( 
\
block_1_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
~
block_1_expand_relu/Relu6Relu6 block_1_expand_BN/FusedBatchNorm*
T0*/
_output_shapes
:���������pp`
�
block_1_pad/Pad/paddingsConst*9
value0B."                               *
dtype0*
_output_shapes

:
�
block_1_pad/PadPadblock_1_expand_relu/Relu6block_1_pad/Pad/paddings*
T0*/
_output_shapes
:���������qq`
�
Cblock_1_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      `      *5
_class+
)'loc:@block_1_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Ablock_1_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *�ȩ�*5
_class+
)'loc:@block_1_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Ablock_1_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *�ȩ=*5
_class+
)'loc:@block_1_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Kblock_1_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformCblock_1_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*5
_class+
)'loc:@block_1_depthwise/depthwise_kernel*
dtype0*&
_output_shapes
:`
�
Ablock_1_depthwise/depthwise_kernel/Initializer/random_uniform/subSubAblock_1_depthwise/depthwise_kernel/Initializer/random_uniform/maxAblock_1_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_1_depthwise/depthwise_kernel*
_output_shapes
: 
�
Ablock_1_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulKblock_1_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformAblock_1_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@block_1_depthwise/depthwise_kernel*&
_output_shapes
:`
�
=block_1_depthwise/depthwise_kernel/Initializer/random_uniformAddAblock_1_depthwise/depthwise_kernel/Initializer/random_uniform/mulAblock_1_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_1_depthwise/depthwise_kernel*&
_output_shapes
:`
�
"block_1_depthwise/depthwise_kernelVarHandleOp*
shape:`*3
shared_name$"block_1_depthwise/depthwise_kernel*5
_class+
)'loc:@block_1_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Cblock_1_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_1_depthwise/depthwise_kernel*
_output_shapes
: 
�
)block_1_depthwise/depthwise_kernel/AssignAssignVariableOp"block_1_depthwise/depthwise_kernel=block_1_depthwise/depthwise_kernel/Initializer/random_uniform*5
_class+
)'loc:@block_1_depthwise/depthwise_kernel*
dtype0
�
6block_1_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp"block_1_depthwise/depthwise_kernel*5
_class+
)'loc:@block_1_depthwise/depthwise_kernel*
dtype0*&
_output_shapes
:`
�
*block_1_depthwise/depthwise/ReadVariableOpReadVariableOp"block_1_depthwise/depthwise_kernel*
dtype0*&
_output_shapes
:`
z
!block_1_depthwise/depthwise/ShapeConst*%
valueB"      `      *
dtype0*
_output_shapes
:
z
)block_1_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_1_depthwise/depthwiseDepthwiseConv2dNativeblock_1_pad/Pad*block_1_depthwise/depthwise/ReadVariableOp*
paddingVALID*
T0*
strides
*/
_output_shapes
:���������88`
�
+block_1_depthwise_BN/gamma/Initializer/onesConst*
valueB`*  �?*-
_class#
!loc:@block_1_depthwise_BN/gamma*
dtype0*
_output_shapes
:`
�
block_1_depthwise_BN/gammaVarHandleOp*
shape:`*+
shared_nameblock_1_depthwise_BN/gamma*-
_class#
!loc:@block_1_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
;block_1_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_1_depthwise_BN/gamma*
_output_shapes
: 
�
!block_1_depthwise_BN/gamma/AssignAssignVariableOpblock_1_depthwise_BN/gamma+block_1_depthwise_BN/gamma/Initializer/ones*-
_class#
!loc:@block_1_depthwise_BN/gamma*
dtype0
�
.block_1_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_1_depthwise_BN/gamma*-
_class#
!loc:@block_1_depthwise_BN/gamma*
dtype0*
_output_shapes
:`
�
+block_1_depthwise_BN/beta/Initializer/zerosConst*
valueB`*    *,
_class"
 loc:@block_1_depthwise_BN/beta*
dtype0*
_output_shapes
:`
�
block_1_depthwise_BN/betaVarHandleOp*
shape:`**
shared_nameblock_1_depthwise_BN/beta*,
_class"
 loc:@block_1_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
:block_1_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_1_depthwise_BN/beta*
_output_shapes
: 
�
 block_1_depthwise_BN/beta/AssignAssignVariableOpblock_1_depthwise_BN/beta+block_1_depthwise_BN/beta/Initializer/zeros*,
_class"
 loc:@block_1_depthwise_BN/beta*
dtype0
�
-block_1_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_1_depthwise_BN/beta*,
_class"
 loc:@block_1_depthwise_BN/beta*
dtype0*
_output_shapes
:`
�
2block_1_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB`*    *3
_class)
'%loc:@block_1_depthwise_BN/moving_mean*
dtype0*
_output_shapes
:`
�
 block_1_depthwise_BN/moving_meanVarHandleOp*
shape:`*1
shared_name" block_1_depthwise_BN/moving_mean*3
_class)
'%loc:@block_1_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Ablock_1_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp block_1_depthwise_BN/moving_mean*
_output_shapes
: 
�
'block_1_depthwise_BN/moving_mean/AssignAssignVariableOp block_1_depthwise_BN/moving_mean2block_1_depthwise_BN/moving_mean/Initializer/zeros*3
_class)
'%loc:@block_1_depthwise_BN/moving_mean*
dtype0
�
4block_1_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp block_1_depthwise_BN/moving_mean*3
_class)
'%loc:@block_1_depthwise_BN/moving_mean*
dtype0*
_output_shapes
:`
�
5block_1_depthwise_BN/moving_variance/Initializer/onesConst*
valueB`*  �?*7
_class-
+)loc:@block_1_depthwise_BN/moving_variance*
dtype0*
_output_shapes
:`
�
$block_1_depthwise_BN/moving_varianceVarHandleOp*
shape:`*5
shared_name&$block_1_depthwise_BN/moving_variance*7
_class-
+)loc:@block_1_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Eblock_1_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp$block_1_depthwise_BN/moving_variance*
_output_shapes
: 
�
+block_1_depthwise_BN/moving_variance/AssignAssignVariableOp$block_1_depthwise_BN/moving_variance5block_1_depthwise_BN/moving_variance/Initializer/ones*7
_class-
+)loc:@block_1_depthwise_BN/moving_variance*
dtype0
�
8block_1_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp$block_1_depthwise_BN/moving_variance*7
_class-
+)loc:@block_1_depthwise_BN/moving_variance*
dtype0*
_output_shapes
:`
z
#block_1_depthwise_BN/ReadVariableOpReadVariableOpblock_1_depthwise_BN/gamma*
dtype0*
_output_shapes
:`
{
%block_1_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_1_depthwise_BN/beta*
dtype0*
_output_shapes
:`
�
2block_1_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp block_1_depthwise_BN/moving_mean*
dtype0*
_output_shapes
:`
�
4block_1_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp$block_1_depthwise_BN/moving_variance*
dtype0*
_output_shapes
:`
�
#block_1_depthwise_BN/FusedBatchNormFusedBatchNormblock_1_depthwise/depthwise#block_1_depthwise_BN/ReadVariableOp%block_1_depthwise_BN/ReadVariableOp_12block_1_depthwise_BN/FusedBatchNorm/ReadVariableOp4block_1_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:���������88`:`:`:`:`*
is_training( 
_
block_1_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_1_depthwise_relu/Relu6Relu6#block_1_depthwise_BN/FusedBatchNorm*
T0*/
_output_shapes
:���������88`
�
7block_1_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      `      *)
_class
loc:@block_1_project/kernel*
dtype0*
_output_shapes
:
�
5block_1_project/kernel/Initializer/random_uniform/minConst*
valueB
 *.�d�*)
_class
loc:@block_1_project/kernel*
dtype0*
_output_shapes
: 
�
5block_1_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *.�d>*)
_class
loc:@block_1_project/kernel*
dtype0*
_output_shapes
: 
�
?block_1_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_1_project/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_1_project/kernel*
dtype0*&
_output_shapes
:`
�
5block_1_project/kernel/Initializer/random_uniform/subSub5block_1_project/kernel/Initializer/random_uniform/max5block_1_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_1_project/kernel*
_output_shapes
: 
�
5block_1_project/kernel/Initializer/random_uniform/mulMul?block_1_project/kernel/Initializer/random_uniform/RandomUniform5block_1_project/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_1_project/kernel*&
_output_shapes
:`
�
1block_1_project/kernel/Initializer/random_uniformAdd5block_1_project/kernel/Initializer/random_uniform/mul5block_1_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_1_project/kernel*&
_output_shapes
:`
�
block_1_project/kernelVarHandleOp*
shape:`*'
shared_nameblock_1_project/kernel*)
_class
loc:@block_1_project/kernel*
dtype0*
_output_shapes
: 
}
7block_1_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_1_project/kernel*
_output_shapes
: 
�
block_1_project/kernel/AssignAssignVariableOpblock_1_project/kernel1block_1_project/kernel/Initializer/random_uniform*)
_class
loc:@block_1_project/kernel*
dtype0
�
*block_1_project/kernel/Read/ReadVariableOpReadVariableOpblock_1_project/kernel*)
_class
loc:@block_1_project/kernel*
dtype0*&
_output_shapes
:`
n
block_1_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_1_project/Conv2D/ReadVariableOpReadVariableOpblock_1_project/kernel*
dtype0*&
_output_shapes
:`
�
block_1_project/Conv2DConv2Dblock_1_depthwise_relu/Relu6%block_1_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:���������88
�
)block_1_project_BN/gamma/Initializer/onesConst*
valueB*  �?*+
_class!
loc:@block_1_project_BN/gamma*
dtype0*
_output_shapes
:
�
block_1_project_BN/gammaVarHandleOp*
shape:*)
shared_nameblock_1_project_BN/gamma*+
_class!
loc:@block_1_project_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_1_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_1_project_BN/gamma*
_output_shapes
: 
�
block_1_project_BN/gamma/AssignAssignVariableOpblock_1_project_BN/gamma)block_1_project_BN/gamma/Initializer/ones*+
_class!
loc:@block_1_project_BN/gamma*
dtype0
�
,block_1_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_1_project_BN/gamma*+
_class!
loc:@block_1_project_BN/gamma*
dtype0*
_output_shapes
:
�
)block_1_project_BN/beta/Initializer/zerosConst*
valueB*    **
_class 
loc:@block_1_project_BN/beta*
dtype0*
_output_shapes
:
�
block_1_project_BN/betaVarHandleOp*
shape:*(
shared_nameblock_1_project_BN/beta**
_class 
loc:@block_1_project_BN/beta*
dtype0*
_output_shapes
: 

8block_1_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_1_project_BN/beta*
_output_shapes
: 
�
block_1_project_BN/beta/AssignAssignVariableOpblock_1_project_BN/beta)block_1_project_BN/beta/Initializer/zeros**
_class 
loc:@block_1_project_BN/beta*
dtype0
�
+block_1_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_1_project_BN/beta**
_class 
loc:@block_1_project_BN/beta*
dtype0*
_output_shapes
:
�
0block_1_project_BN/moving_mean/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@block_1_project_BN/moving_mean*
dtype0*
_output_shapes
:
�
block_1_project_BN/moving_meanVarHandleOp*
shape:*/
shared_name block_1_project_BN/moving_mean*1
_class'
%#loc:@block_1_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_1_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_1_project_BN/moving_mean*
_output_shapes
: 
�
%block_1_project_BN/moving_mean/AssignAssignVariableOpblock_1_project_BN/moving_mean0block_1_project_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_1_project_BN/moving_mean*
dtype0
�
2block_1_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_1_project_BN/moving_mean*1
_class'
%#loc:@block_1_project_BN/moving_mean*
dtype0*
_output_shapes
:
�
3block_1_project_BN/moving_variance/Initializer/onesConst*
valueB*  �?*5
_class+
)'loc:@block_1_project_BN/moving_variance*
dtype0*
_output_shapes
:
�
"block_1_project_BN/moving_varianceVarHandleOp*
shape:*3
shared_name$"block_1_project_BN/moving_variance*5
_class+
)'loc:@block_1_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_1_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_1_project_BN/moving_variance*
_output_shapes
: 
�
)block_1_project_BN/moving_variance/AssignAssignVariableOp"block_1_project_BN/moving_variance3block_1_project_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_1_project_BN/moving_variance*
dtype0
�
6block_1_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_1_project_BN/moving_variance*5
_class+
)'loc:@block_1_project_BN/moving_variance*
dtype0*
_output_shapes
:
v
!block_1_project_BN/ReadVariableOpReadVariableOpblock_1_project_BN/gamma*
dtype0*
_output_shapes
:
w
#block_1_project_BN/ReadVariableOp_1ReadVariableOpblock_1_project_BN/beta*
dtype0*
_output_shapes
:
�
0block_1_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_1_project_BN/moving_mean*
dtype0*
_output_shapes
:
�
2block_1_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_1_project_BN/moving_variance*
dtype0*
_output_shapes
:
�
!block_1_project_BN/FusedBatchNormFusedBatchNormblock_1_project/Conv2D!block_1_project_BN/ReadVariableOp#block_1_project_BN/ReadVariableOp_10block_1_project_BN/FusedBatchNorm/ReadVariableOp2block_1_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:���������88::::*
is_training( 
]
block_1_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
6block_2_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"         �   *(
_class
loc:@block_2_expand/kernel*
dtype0*
_output_shapes
:
�
4block_2_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *��A�*(
_class
loc:@block_2_expand/kernel*
dtype0*
_output_shapes
: 
�
4block_2_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *��A>*(
_class
loc:@block_2_expand/kernel*
dtype0*
_output_shapes
: 
�
>block_2_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform6block_2_expand/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@block_2_expand/kernel*
dtype0*'
_output_shapes
:�
�
4block_2_expand/kernel/Initializer/random_uniform/subSub4block_2_expand/kernel/Initializer/random_uniform/max4block_2_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_2_expand/kernel*
_output_shapes
: 
�
4block_2_expand/kernel/Initializer/random_uniform/mulMul>block_2_expand/kernel/Initializer/random_uniform/RandomUniform4block_2_expand/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@block_2_expand/kernel*'
_output_shapes
:�
�
0block_2_expand/kernel/Initializer/random_uniformAdd4block_2_expand/kernel/Initializer/random_uniform/mul4block_2_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_2_expand/kernel*'
_output_shapes
:�
�
block_2_expand/kernelVarHandleOp*
shape:�*&
shared_nameblock_2_expand/kernel*(
_class
loc:@block_2_expand/kernel*
dtype0*
_output_shapes
: 
{
6block_2_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_2_expand/kernel*
_output_shapes
: 
�
block_2_expand/kernel/AssignAssignVariableOpblock_2_expand/kernel0block_2_expand/kernel/Initializer/random_uniform*(
_class
loc:@block_2_expand/kernel*
dtype0
�
)block_2_expand/kernel/Read/ReadVariableOpReadVariableOpblock_2_expand/kernel*(
_class
loc:@block_2_expand/kernel*
dtype0*'
_output_shapes
:�
m
block_2_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
$block_2_expand/Conv2D/ReadVariableOpReadVariableOpblock_2_expand/kernel*
dtype0*'
_output_shapes
:�
�
block_2_expand/Conv2DConv2D!block_1_project_BN/FusedBatchNorm$block_2_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:���������88�
�
(block_2_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?**
_class 
loc:@block_2_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_2_expand_BN/gammaVarHandleOp*
shape:�*(
shared_nameblock_2_expand_BN/gamma**
_class 
loc:@block_2_expand_BN/gamma*
dtype0*
_output_shapes
: 

8block_2_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_2_expand_BN/gamma*
_output_shapes
: 
�
block_2_expand_BN/gamma/AssignAssignVariableOpblock_2_expand_BN/gamma(block_2_expand_BN/gamma/Initializer/ones**
_class 
loc:@block_2_expand_BN/gamma*
dtype0
�
+block_2_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_2_expand_BN/gamma**
_class 
loc:@block_2_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
(block_2_expand_BN/beta/Initializer/zerosConst*
valueB�*    *)
_class
loc:@block_2_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_2_expand_BN/betaVarHandleOp*
shape:�*'
shared_nameblock_2_expand_BN/beta*)
_class
loc:@block_2_expand_BN/beta*
dtype0*
_output_shapes
: 
}
7block_2_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_2_expand_BN/beta*
_output_shapes
: 
�
block_2_expand_BN/beta/AssignAssignVariableOpblock_2_expand_BN/beta(block_2_expand_BN/beta/Initializer/zeros*)
_class
loc:@block_2_expand_BN/beta*
dtype0
�
*block_2_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_2_expand_BN/beta*)
_class
loc:@block_2_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_2_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *0
_class&
$"loc:@block_2_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_2_expand_BN/moving_meanVarHandleOp*
shape:�*.
shared_nameblock_2_expand_BN/moving_mean*0
_class&
$"loc:@block_2_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
>block_2_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_2_expand_BN/moving_mean*
_output_shapes
: 
�
$block_2_expand_BN/moving_mean/AssignAssignVariableOpblock_2_expand_BN/moving_mean/block_2_expand_BN/moving_mean/Initializer/zeros*0
_class&
$"loc:@block_2_expand_BN/moving_mean*
dtype0
�
1block_2_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_2_expand_BN/moving_mean*0
_class&
$"loc:@block_2_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_2_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*4
_class*
(&loc:@block_2_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_2_expand_BN/moving_varianceVarHandleOp*
shape:�*2
shared_name#!block_2_expand_BN/moving_variance*4
_class*
(&loc:@block_2_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Bblock_2_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_2_expand_BN/moving_variance*
_output_shapes
: 
�
(block_2_expand_BN/moving_variance/AssignAssignVariableOp!block_2_expand_BN/moving_variance2block_2_expand_BN/moving_variance/Initializer/ones*4
_class*
(&loc:@block_2_expand_BN/moving_variance*
dtype0
�
5block_2_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp!block_2_expand_BN/moving_variance*4
_class*
(&loc:@block_2_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
u
 block_2_expand_BN/ReadVariableOpReadVariableOpblock_2_expand_BN/gamma*
dtype0*
_output_shapes	
:�
v
"block_2_expand_BN/ReadVariableOp_1ReadVariableOpblock_2_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_2_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_2_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
1block_2_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp!block_2_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
 block_2_expand_BN/FusedBatchNormFusedBatchNormblock_2_expand/Conv2D block_2_expand_BN/ReadVariableOp"block_2_expand_BN/ReadVariableOp_1/block_2_expand_BN/FusedBatchNorm/ReadVariableOp1block_2_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:���������88�:�:�:�:�*
is_training( 
\
block_2_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 

block_2_expand_relu/Relu6Relu6 block_2_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:���������88�
�
Cblock_2_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      �      *5
_class+
)'loc:@block_2_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Ablock_2_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *ފ�*5
_class+
)'loc:@block_2_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Ablock_2_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *ފ=*5
_class+
)'loc:@block_2_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Kblock_2_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformCblock_2_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*5
_class+
)'loc:@block_2_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Ablock_2_depthwise/depthwise_kernel/Initializer/random_uniform/subSubAblock_2_depthwise/depthwise_kernel/Initializer/random_uniform/maxAblock_2_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_2_depthwise/depthwise_kernel*
_output_shapes
: 
�
Ablock_2_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulKblock_2_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformAblock_2_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@block_2_depthwise/depthwise_kernel*'
_output_shapes
:�
�
=block_2_depthwise/depthwise_kernel/Initializer/random_uniformAddAblock_2_depthwise/depthwise_kernel/Initializer/random_uniform/mulAblock_2_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_2_depthwise/depthwise_kernel*'
_output_shapes
:�
�
"block_2_depthwise/depthwise_kernelVarHandleOp*
shape:�*3
shared_name$"block_2_depthwise/depthwise_kernel*5
_class+
)'loc:@block_2_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Cblock_2_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_2_depthwise/depthwise_kernel*
_output_shapes
: 
�
)block_2_depthwise/depthwise_kernel/AssignAssignVariableOp"block_2_depthwise/depthwise_kernel=block_2_depthwise/depthwise_kernel/Initializer/random_uniform*5
_class+
)'loc:@block_2_depthwise/depthwise_kernel*
dtype0
�
6block_2_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp"block_2_depthwise/depthwise_kernel*5
_class+
)'loc:@block_2_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
*block_2_depthwise/depthwise/ReadVariableOpReadVariableOp"block_2_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
z
!block_2_depthwise/depthwise/ShapeConst*%
valueB"      �      *
dtype0*
_output_shapes
:
z
)block_2_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_2_depthwise/depthwiseDepthwiseConv2dNativeblock_2_expand_relu/Relu6*block_2_depthwise/depthwise/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:���������88�
�
+block_2_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*-
_class#
!loc:@block_2_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_2_depthwise_BN/gammaVarHandleOp*
shape:�*+
shared_nameblock_2_depthwise_BN/gamma*-
_class#
!loc:@block_2_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
;block_2_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_2_depthwise_BN/gamma*
_output_shapes
: 
�
!block_2_depthwise_BN/gamma/AssignAssignVariableOpblock_2_depthwise_BN/gamma+block_2_depthwise_BN/gamma/Initializer/ones*-
_class#
!loc:@block_2_depthwise_BN/gamma*
dtype0
�
.block_2_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_2_depthwise_BN/gamma*-
_class#
!loc:@block_2_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
+block_2_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *,
_class"
 loc:@block_2_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_2_depthwise_BN/betaVarHandleOp*
shape:�**
shared_nameblock_2_depthwise_BN/beta*,
_class"
 loc:@block_2_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
:block_2_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_2_depthwise_BN/beta*
_output_shapes
: 
�
 block_2_depthwise_BN/beta/AssignAssignVariableOpblock_2_depthwise_BN/beta+block_2_depthwise_BN/beta/Initializer/zeros*,
_class"
 loc:@block_2_depthwise_BN/beta*
dtype0
�
-block_2_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_2_depthwise_BN/beta*,
_class"
 loc:@block_2_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_2_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *3
_class)
'%loc:@block_2_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
 block_2_depthwise_BN/moving_meanVarHandleOp*
shape:�*1
shared_name" block_2_depthwise_BN/moving_mean*3
_class)
'%loc:@block_2_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Ablock_2_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp block_2_depthwise_BN/moving_mean*
_output_shapes
: 
�
'block_2_depthwise_BN/moving_mean/AssignAssignVariableOp block_2_depthwise_BN/moving_mean2block_2_depthwise_BN/moving_mean/Initializer/zeros*3
_class)
'%loc:@block_2_depthwise_BN/moving_mean*
dtype0
�
4block_2_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp block_2_depthwise_BN/moving_mean*3
_class)
'%loc:@block_2_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_2_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*7
_class-
+)loc:@block_2_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_2_depthwise_BN/moving_varianceVarHandleOp*
shape:�*5
shared_name&$block_2_depthwise_BN/moving_variance*7
_class-
+)loc:@block_2_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Eblock_2_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp$block_2_depthwise_BN/moving_variance*
_output_shapes
: 
�
+block_2_depthwise_BN/moving_variance/AssignAssignVariableOp$block_2_depthwise_BN/moving_variance5block_2_depthwise_BN/moving_variance/Initializer/ones*7
_class-
+)loc:@block_2_depthwise_BN/moving_variance*
dtype0
�
8block_2_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp$block_2_depthwise_BN/moving_variance*7
_class-
+)loc:@block_2_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
{
#block_2_depthwise_BN/ReadVariableOpReadVariableOpblock_2_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
|
%block_2_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_2_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_2_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp block_2_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
4block_2_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp$block_2_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
#block_2_depthwise_BN/FusedBatchNormFusedBatchNormblock_2_depthwise/depthwise#block_2_depthwise_BN/ReadVariableOp%block_2_depthwise_BN/ReadVariableOp_12block_2_depthwise_BN/FusedBatchNorm/ReadVariableOp4block_2_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:���������88�:�:�:�:�*
is_training( 
_
block_2_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_2_depthwise_relu/Relu6Relu6#block_2_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:���������88�
�
7block_2_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �      *)
_class
loc:@block_2_project/kernel*
dtype0*
_output_shapes
:
�
5block_2_project/kernel/Initializer/random_uniform/minConst*
valueB
 *��A�*)
_class
loc:@block_2_project/kernel*
dtype0*
_output_shapes
: 
�
5block_2_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *��A>*)
_class
loc:@block_2_project/kernel*
dtype0*
_output_shapes
: 
�
?block_2_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_2_project/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_2_project/kernel*
dtype0*'
_output_shapes
:�
�
5block_2_project/kernel/Initializer/random_uniform/subSub5block_2_project/kernel/Initializer/random_uniform/max5block_2_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_2_project/kernel*
_output_shapes
: 
�
5block_2_project/kernel/Initializer/random_uniform/mulMul?block_2_project/kernel/Initializer/random_uniform/RandomUniform5block_2_project/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_2_project/kernel*'
_output_shapes
:�
�
1block_2_project/kernel/Initializer/random_uniformAdd5block_2_project/kernel/Initializer/random_uniform/mul5block_2_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_2_project/kernel*'
_output_shapes
:�
�
block_2_project/kernelVarHandleOp*
shape:�*'
shared_nameblock_2_project/kernel*)
_class
loc:@block_2_project/kernel*
dtype0*
_output_shapes
: 
}
7block_2_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_2_project/kernel*
_output_shapes
: 
�
block_2_project/kernel/AssignAssignVariableOpblock_2_project/kernel1block_2_project/kernel/Initializer/random_uniform*)
_class
loc:@block_2_project/kernel*
dtype0
�
*block_2_project/kernel/Read/ReadVariableOpReadVariableOpblock_2_project/kernel*)
_class
loc:@block_2_project/kernel*
dtype0*'
_output_shapes
:�
n
block_2_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_2_project/Conv2D/ReadVariableOpReadVariableOpblock_2_project/kernel*
dtype0*'
_output_shapes
:�
�
block_2_project/Conv2DConv2Dblock_2_depthwise_relu/Relu6%block_2_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:���������88
�
)block_2_project_BN/gamma/Initializer/onesConst*
valueB*  �?*+
_class!
loc:@block_2_project_BN/gamma*
dtype0*
_output_shapes
:
�
block_2_project_BN/gammaVarHandleOp*
shape:*)
shared_nameblock_2_project_BN/gamma*+
_class!
loc:@block_2_project_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_2_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_2_project_BN/gamma*
_output_shapes
: 
�
block_2_project_BN/gamma/AssignAssignVariableOpblock_2_project_BN/gamma)block_2_project_BN/gamma/Initializer/ones*+
_class!
loc:@block_2_project_BN/gamma*
dtype0
�
,block_2_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_2_project_BN/gamma*+
_class!
loc:@block_2_project_BN/gamma*
dtype0*
_output_shapes
:
�
)block_2_project_BN/beta/Initializer/zerosConst*
valueB*    **
_class 
loc:@block_2_project_BN/beta*
dtype0*
_output_shapes
:
�
block_2_project_BN/betaVarHandleOp*
shape:*(
shared_nameblock_2_project_BN/beta**
_class 
loc:@block_2_project_BN/beta*
dtype0*
_output_shapes
: 

8block_2_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_2_project_BN/beta*
_output_shapes
: 
�
block_2_project_BN/beta/AssignAssignVariableOpblock_2_project_BN/beta)block_2_project_BN/beta/Initializer/zeros**
_class 
loc:@block_2_project_BN/beta*
dtype0
�
+block_2_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_2_project_BN/beta**
_class 
loc:@block_2_project_BN/beta*
dtype0*
_output_shapes
:
�
0block_2_project_BN/moving_mean/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@block_2_project_BN/moving_mean*
dtype0*
_output_shapes
:
�
block_2_project_BN/moving_meanVarHandleOp*
shape:*/
shared_name block_2_project_BN/moving_mean*1
_class'
%#loc:@block_2_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_2_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_2_project_BN/moving_mean*
_output_shapes
: 
�
%block_2_project_BN/moving_mean/AssignAssignVariableOpblock_2_project_BN/moving_mean0block_2_project_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_2_project_BN/moving_mean*
dtype0
�
2block_2_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_2_project_BN/moving_mean*1
_class'
%#loc:@block_2_project_BN/moving_mean*
dtype0*
_output_shapes
:
�
3block_2_project_BN/moving_variance/Initializer/onesConst*
valueB*  �?*5
_class+
)'loc:@block_2_project_BN/moving_variance*
dtype0*
_output_shapes
:
�
"block_2_project_BN/moving_varianceVarHandleOp*
shape:*3
shared_name$"block_2_project_BN/moving_variance*5
_class+
)'loc:@block_2_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_2_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_2_project_BN/moving_variance*
_output_shapes
: 
�
)block_2_project_BN/moving_variance/AssignAssignVariableOp"block_2_project_BN/moving_variance3block_2_project_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_2_project_BN/moving_variance*
dtype0
�
6block_2_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_2_project_BN/moving_variance*5
_class+
)'loc:@block_2_project_BN/moving_variance*
dtype0*
_output_shapes
:
v
!block_2_project_BN/ReadVariableOpReadVariableOpblock_2_project_BN/gamma*
dtype0*
_output_shapes
:
w
#block_2_project_BN/ReadVariableOp_1ReadVariableOpblock_2_project_BN/beta*
dtype0*
_output_shapes
:
�
0block_2_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_2_project_BN/moving_mean*
dtype0*
_output_shapes
:
�
2block_2_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_2_project_BN/moving_variance*
dtype0*
_output_shapes
:
�
!block_2_project_BN/FusedBatchNormFusedBatchNormblock_2_project/Conv2D!block_2_project_BN/ReadVariableOp#block_2_project_BN/ReadVariableOp_10block_2_project_BN/FusedBatchNorm/ReadVariableOp2block_2_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:���������88::::*
is_training( 
]
block_2_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_2_add/addAdd!block_1_project_BN/FusedBatchNorm!block_2_project_BN/FusedBatchNorm*
T0*/
_output_shapes
:���������88
�
6block_3_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"         �   *(
_class
loc:@block_3_expand/kernel*
dtype0*
_output_shapes
:
�
4block_3_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *��A�*(
_class
loc:@block_3_expand/kernel*
dtype0*
_output_shapes
: 
�
4block_3_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *��A>*(
_class
loc:@block_3_expand/kernel*
dtype0*
_output_shapes
: 
�
>block_3_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform6block_3_expand/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@block_3_expand/kernel*
dtype0*'
_output_shapes
:�
�
4block_3_expand/kernel/Initializer/random_uniform/subSub4block_3_expand/kernel/Initializer/random_uniform/max4block_3_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_3_expand/kernel*
_output_shapes
: 
�
4block_3_expand/kernel/Initializer/random_uniform/mulMul>block_3_expand/kernel/Initializer/random_uniform/RandomUniform4block_3_expand/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@block_3_expand/kernel*'
_output_shapes
:�
�
0block_3_expand/kernel/Initializer/random_uniformAdd4block_3_expand/kernel/Initializer/random_uniform/mul4block_3_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_3_expand/kernel*'
_output_shapes
:�
�
block_3_expand/kernelVarHandleOp*
shape:�*&
shared_nameblock_3_expand/kernel*(
_class
loc:@block_3_expand/kernel*
dtype0*
_output_shapes
: 
{
6block_3_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_3_expand/kernel*
_output_shapes
: 
�
block_3_expand/kernel/AssignAssignVariableOpblock_3_expand/kernel0block_3_expand/kernel/Initializer/random_uniform*(
_class
loc:@block_3_expand/kernel*
dtype0
�
)block_3_expand/kernel/Read/ReadVariableOpReadVariableOpblock_3_expand/kernel*(
_class
loc:@block_3_expand/kernel*
dtype0*'
_output_shapes
:�
m
block_3_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
$block_3_expand/Conv2D/ReadVariableOpReadVariableOpblock_3_expand/kernel*
dtype0*'
_output_shapes
:�
�
block_3_expand/Conv2DConv2Dblock_2_add/add$block_3_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:���������88�
�
(block_3_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?**
_class 
loc:@block_3_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_3_expand_BN/gammaVarHandleOp*
shape:�*(
shared_nameblock_3_expand_BN/gamma**
_class 
loc:@block_3_expand_BN/gamma*
dtype0*
_output_shapes
: 

8block_3_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_3_expand_BN/gamma*
_output_shapes
: 
�
block_3_expand_BN/gamma/AssignAssignVariableOpblock_3_expand_BN/gamma(block_3_expand_BN/gamma/Initializer/ones**
_class 
loc:@block_3_expand_BN/gamma*
dtype0
�
+block_3_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_3_expand_BN/gamma**
_class 
loc:@block_3_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
(block_3_expand_BN/beta/Initializer/zerosConst*
valueB�*    *)
_class
loc:@block_3_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_3_expand_BN/betaVarHandleOp*
shape:�*'
shared_nameblock_3_expand_BN/beta*)
_class
loc:@block_3_expand_BN/beta*
dtype0*
_output_shapes
: 
}
7block_3_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_3_expand_BN/beta*
_output_shapes
: 
�
block_3_expand_BN/beta/AssignAssignVariableOpblock_3_expand_BN/beta(block_3_expand_BN/beta/Initializer/zeros*)
_class
loc:@block_3_expand_BN/beta*
dtype0
�
*block_3_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_3_expand_BN/beta*)
_class
loc:@block_3_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_3_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *0
_class&
$"loc:@block_3_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_3_expand_BN/moving_meanVarHandleOp*
shape:�*.
shared_nameblock_3_expand_BN/moving_mean*0
_class&
$"loc:@block_3_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
>block_3_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_3_expand_BN/moving_mean*
_output_shapes
: 
�
$block_3_expand_BN/moving_mean/AssignAssignVariableOpblock_3_expand_BN/moving_mean/block_3_expand_BN/moving_mean/Initializer/zeros*0
_class&
$"loc:@block_3_expand_BN/moving_mean*
dtype0
�
1block_3_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_3_expand_BN/moving_mean*0
_class&
$"loc:@block_3_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_3_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*4
_class*
(&loc:@block_3_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_3_expand_BN/moving_varianceVarHandleOp*
shape:�*2
shared_name#!block_3_expand_BN/moving_variance*4
_class*
(&loc:@block_3_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Bblock_3_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_3_expand_BN/moving_variance*
_output_shapes
: 
�
(block_3_expand_BN/moving_variance/AssignAssignVariableOp!block_3_expand_BN/moving_variance2block_3_expand_BN/moving_variance/Initializer/ones*4
_class*
(&loc:@block_3_expand_BN/moving_variance*
dtype0
�
5block_3_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp!block_3_expand_BN/moving_variance*4
_class*
(&loc:@block_3_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
u
 block_3_expand_BN/ReadVariableOpReadVariableOpblock_3_expand_BN/gamma*
dtype0*
_output_shapes	
:�
v
"block_3_expand_BN/ReadVariableOp_1ReadVariableOpblock_3_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_3_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_3_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
1block_3_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp!block_3_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
 block_3_expand_BN/FusedBatchNormFusedBatchNormblock_3_expand/Conv2D block_3_expand_BN/ReadVariableOp"block_3_expand_BN/ReadVariableOp_1/block_3_expand_BN/FusedBatchNorm/ReadVariableOp1block_3_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:���������88�:�:�:�:�*
is_training( 
\
block_3_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 

block_3_expand_relu/Relu6Relu6 block_3_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:���������88�
�
block_3_pad/Pad/paddingsConst*9
value0B."                               *
dtype0*
_output_shapes

:
�
block_3_pad/PadPadblock_3_expand_relu/Relu6block_3_pad/Pad/paddings*
T0*0
_output_shapes
:���������99�
�
Cblock_3_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      �      *5
_class+
)'loc:@block_3_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Ablock_3_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *ފ�*5
_class+
)'loc:@block_3_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Ablock_3_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *ފ=*5
_class+
)'loc:@block_3_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Kblock_3_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformCblock_3_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*5
_class+
)'loc:@block_3_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Ablock_3_depthwise/depthwise_kernel/Initializer/random_uniform/subSubAblock_3_depthwise/depthwise_kernel/Initializer/random_uniform/maxAblock_3_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_3_depthwise/depthwise_kernel*
_output_shapes
: 
�
Ablock_3_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulKblock_3_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformAblock_3_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@block_3_depthwise/depthwise_kernel*'
_output_shapes
:�
�
=block_3_depthwise/depthwise_kernel/Initializer/random_uniformAddAblock_3_depthwise/depthwise_kernel/Initializer/random_uniform/mulAblock_3_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_3_depthwise/depthwise_kernel*'
_output_shapes
:�
�
"block_3_depthwise/depthwise_kernelVarHandleOp*
shape:�*3
shared_name$"block_3_depthwise/depthwise_kernel*5
_class+
)'loc:@block_3_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Cblock_3_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_3_depthwise/depthwise_kernel*
_output_shapes
: 
�
)block_3_depthwise/depthwise_kernel/AssignAssignVariableOp"block_3_depthwise/depthwise_kernel=block_3_depthwise/depthwise_kernel/Initializer/random_uniform*5
_class+
)'loc:@block_3_depthwise/depthwise_kernel*
dtype0
�
6block_3_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp"block_3_depthwise/depthwise_kernel*5
_class+
)'loc:@block_3_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
*block_3_depthwise/depthwise/ReadVariableOpReadVariableOp"block_3_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
z
!block_3_depthwise/depthwise/ShapeConst*%
valueB"      �      *
dtype0*
_output_shapes
:
z
)block_3_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_3_depthwise/depthwiseDepthwiseConv2dNativeblock_3_pad/Pad*block_3_depthwise/depthwise/ReadVariableOp*
paddingVALID*
T0*
strides
*0
_output_shapes
:����������
�
+block_3_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*-
_class#
!loc:@block_3_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_3_depthwise_BN/gammaVarHandleOp*
shape:�*+
shared_nameblock_3_depthwise_BN/gamma*-
_class#
!loc:@block_3_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
;block_3_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_3_depthwise_BN/gamma*
_output_shapes
: 
�
!block_3_depthwise_BN/gamma/AssignAssignVariableOpblock_3_depthwise_BN/gamma+block_3_depthwise_BN/gamma/Initializer/ones*-
_class#
!loc:@block_3_depthwise_BN/gamma*
dtype0
�
.block_3_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_3_depthwise_BN/gamma*-
_class#
!loc:@block_3_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
+block_3_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *,
_class"
 loc:@block_3_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_3_depthwise_BN/betaVarHandleOp*
shape:�**
shared_nameblock_3_depthwise_BN/beta*,
_class"
 loc:@block_3_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
:block_3_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_3_depthwise_BN/beta*
_output_shapes
: 
�
 block_3_depthwise_BN/beta/AssignAssignVariableOpblock_3_depthwise_BN/beta+block_3_depthwise_BN/beta/Initializer/zeros*,
_class"
 loc:@block_3_depthwise_BN/beta*
dtype0
�
-block_3_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_3_depthwise_BN/beta*,
_class"
 loc:@block_3_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_3_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *3
_class)
'%loc:@block_3_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
 block_3_depthwise_BN/moving_meanVarHandleOp*
shape:�*1
shared_name" block_3_depthwise_BN/moving_mean*3
_class)
'%loc:@block_3_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Ablock_3_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp block_3_depthwise_BN/moving_mean*
_output_shapes
: 
�
'block_3_depthwise_BN/moving_mean/AssignAssignVariableOp block_3_depthwise_BN/moving_mean2block_3_depthwise_BN/moving_mean/Initializer/zeros*3
_class)
'%loc:@block_3_depthwise_BN/moving_mean*
dtype0
�
4block_3_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp block_3_depthwise_BN/moving_mean*3
_class)
'%loc:@block_3_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_3_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*7
_class-
+)loc:@block_3_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_3_depthwise_BN/moving_varianceVarHandleOp*
shape:�*5
shared_name&$block_3_depthwise_BN/moving_variance*7
_class-
+)loc:@block_3_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Eblock_3_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp$block_3_depthwise_BN/moving_variance*
_output_shapes
: 
�
+block_3_depthwise_BN/moving_variance/AssignAssignVariableOp$block_3_depthwise_BN/moving_variance5block_3_depthwise_BN/moving_variance/Initializer/ones*7
_class-
+)loc:@block_3_depthwise_BN/moving_variance*
dtype0
�
8block_3_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp$block_3_depthwise_BN/moving_variance*7
_class-
+)loc:@block_3_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
{
#block_3_depthwise_BN/ReadVariableOpReadVariableOpblock_3_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
|
%block_3_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_3_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_3_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp block_3_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
4block_3_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp$block_3_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
#block_3_depthwise_BN/FusedBatchNormFusedBatchNormblock_3_depthwise/depthwise#block_3_depthwise_BN/ReadVariableOp%block_3_depthwise_BN/ReadVariableOp_12block_3_depthwise_BN/FusedBatchNorm/ReadVariableOp4block_3_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
_
block_3_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_3_depthwise_relu/Relu6Relu6#block_3_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
7block_3_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �       *)
_class
loc:@block_3_project/kernel*
dtype0*
_output_shapes
:
�
5block_3_project/kernel/Initializer/random_uniform/minConst*
valueB
 *�=�*)
_class
loc:@block_3_project/kernel*
dtype0*
_output_shapes
: 
�
5block_3_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *�=>*)
_class
loc:@block_3_project/kernel*
dtype0*
_output_shapes
: 
�
?block_3_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_3_project/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_3_project/kernel*
dtype0*'
_output_shapes
:� 
�
5block_3_project/kernel/Initializer/random_uniform/subSub5block_3_project/kernel/Initializer/random_uniform/max5block_3_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_3_project/kernel*
_output_shapes
: 
�
5block_3_project/kernel/Initializer/random_uniform/mulMul?block_3_project/kernel/Initializer/random_uniform/RandomUniform5block_3_project/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_3_project/kernel*'
_output_shapes
:� 
�
1block_3_project/kernel/Initializer/random_uniformAdd5block_3_project/kernel/Initializer/random_uniform/mul5block_3_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_3_project/kernel*'
_output_shapes
:� 
�
block_3_project/kernelVarHandleOp*
shape:� *'
shared_nameblock_3_project/kernel*)
_class
loc:@block_3_project/kernel*
dtype0*
_output_shapes
: 
}
7block_3_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_3_project/kernel*
_output_shapes
: 
�
block_3_project/kernel/AssignAssignVariableOpblock_3_project/kernel1block_3_project/kernel/Initializer/random_uniform*)
_class
loc:@block_3_project/kernel*
dtype0
�
*block_3_project/kernel/Read/ReadVariableOpReadVariableOpblock_3_project/kernel*)
_class
loc:@block_3_project/kernel*
dtype0*'
_output_shapes
:� 
n
block_3_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_3_project/Conv2D/ReadVariableOpReadVariableOpblock_3_project/kernel*
dtype0*'
_output_shapes
:� 
�
block_3_project/Conv2DConv2Dblock_3_depthwise_relu/Relu6%block_3_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:��������� 
�
)block_3_project_BN/gamma/Initializer/onesConst*
valueB *  �?*+
_class!
loc:@block_3_project_BN/gamma*
dtype0*
_output_shapes
: 
�
block_3_project_BN/gammaVarHandleOp*
shape: *)
shared_nameblock_3_project_BN/gamma*+
_class!
loc:@block_3_project_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_3_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_3_project_BN/gamma*
_output_shapes
: 
�
block_3_project_BN/gamma/AssignAssignVariableOpblock_3_project_BN/gamma)block_3_project_BN/gamma/Initializer/ones*+
_class!
loc:@block_3_project_BN/gamma*
dtype0
�
,block_3_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_3_project_BN/gamma*+
_class!
loc:@block_3_project_BN/gamma*
dtype0*
_output_shapes
: 
�
)block_3_project_BN/beta/Initializer/zerosConst*
valueB *    **
_class 
loc:@block_3_project_BN/beta*
dtype0*
_output_shapes
: 
�
block_3_project_BN/betaVarHandleOp*
shape: *(
shared_nameblock_3_project_BN/beta**
_class 
loc:@block_3_project_BN/beta*
dtype0*
_output_shapes
: 

8block_3_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_3_project_BN/beta*
_output_shapes
: 
�
block_3_project_BN/beta/AssignAssignVariableOpblock_3_project_BN/beta)block_3_project_BN/beta/Initializer/zeros**
_class 
loc:@block_3_project_BN/beta*
dtype0
�
+block_3_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_3_project_BN/beta**
_class 
loc:@block_3_project_BN/beta*
dtype0*
_output_shapes
: 
�
0block_3_project_BN/moving_mean/Initializer/zerosConst*
valueB *    *1
_class'
%#loc:@block_3_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
block_3_project_BN/moving_meanVarHandleOp*
shape: */
shared_name block_3_project_BN/moving_mean*1
_class'
%#loc:@block_3_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_3_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_3_project_BN/moving_mean*
_output_shapes
: 
�
%block_3_project_BN/moving_mean/AssignAssignVariableOpblock_3_project_BN/moving_mean0block_3_project_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_3_project_BN/moving_mean*
dtype0
�
2block_3_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_3_project_BN/moving_mean*1
_class'
%#loc:@block_3_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
3block_3_project_BN/moving_variance/Initializer/onesConst*
valueB *  �?*5
_class+
)'loc:@block_3_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
"block_3_project_BN/moving_varianceVarHandleOp*
shape: *3
shared_name$"block_3_project_BN/moving_variance*5
_class+
)'loc:@block_3_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_3_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_3_project_BN/moving_variance*
_output_shapes
: 
�
)block_3_project_BN/moving_variance/AssignAssignVariableOp"block_3_project_BN/moving_variance3block_3_project_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_3_project_BN/moving_variance*
dtype0
�
6block_3_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_3_project_BN/moving_variance*5
_class+
)'loc:@block_3_project_BN/moving_variance*
dtype0*
_output_shapes
: 
v
!block_3_project_BN/ReadVariableOpReadVariableOpblock_3_project_BN/gamma*
dtype0*
_output_shapes
: 
w
#block_3_project_BN/ReadVariableOp_1ReadVariableOpblock_3_project_BN/beta*
dtype0*
_output_shapes
: 
�
0block_3_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_3_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
2block_3_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_3_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
!block_3_project_BN/FusedBatchNormFusedBatchNormblock_3_project/Conv2D!block_3_project_BN/ReadVariableOp#block_3_project_BN/ReadVariableOp_10block_3_project_BN/FusedBatchNorm/ReadVariableOp2block_3_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:��������� : : : : *
is_training( 
]
block_3_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
6block_4_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"          �   *(
_class
loc:@block_4_expand/kernel*
dtype0*
_output_shapes
:
�
4block_4_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *b�'�*(
_class
loc:@block_4_expand/kernel*
dtype0*
_output_shapes
: 
�
4block_4_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *b�'>*(
_class
loc:@block_4_expand/kernel*
dtype0*
_output_shapes
: 
�
>block_4_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform6block_4_expand/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@block_4_expand/kernel*
dtype0*'
_output_shapes
: �
�
4block_4_expand/kernel/Initializer/random_uniform/subSub4block_4_expand/kernel/Initializer/random_uniform/max4block_4_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_4_expand/kernel*
_output_shapes
: 
�
4block_4_expand/kernel/Initializer/random_uniform/mulMul>block_4_expand/kernel/Initializer/random_uniform/RandomUniform4block_4_expand/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@block_4_expand/kernel*'
_output_shapes
: �
�
0block_4_expand/kernel/Initializer/random_uniformAdd4block_4_expand/kernel/Initializer/random_uniform/mul4block_4_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_4_expand/kernel*'
_output_shapes
: �
�
block_4_expand/kernelVarHandleOp*
shape: �*&
shared_nameblock_4_expand/kernel*(
_class
loc:@block_4_expand/kernel*
dtype0*
_output_shapes
: 
{
6block_4_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_4_expand/kernel*
_output_shapes
: 
�
block_4_expand/kernel/AssignAssignVariableOpblock_4_expand/kernel0block_4_expand/kernel/Initializer/random_uniform*(
_class
loc:@block_4_expand/kernel*
dtype0
�
)block_4_expand/kernel/Read/ReadVariableOpReadVariableOpblock_4_expand/kernel*(
_class
loc:@block_4_expand/kernel*
dtype0*'
_output_shapes
: �
m
block_4_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
$block_4_expand/Conv2D/ReadVariableOpReadVariableOpblock_4_expand/kernel*
dtype0*'
_output_shapes
: �
�
block_4_expand/Conv2DConv2D!block_3_project_BN/FusedBatchNorm$block_4_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
(block_4_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?**
_class 
loc:@block_4_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_4_expand_BN/gammaVarHandleOp*
shape:�*(
shared_nameblock_4_expand_BN/gamma**
_class 
loc:@block_4_expand_BN/gamma*
dtype0*
_output_shapes
: 

8block_4_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_4_expand_BN/gamma*
_output_shapes
: 
�
block_4_expand_BN/gamma/AssignAssignVariableOpblock_4_expand_BN/gamma(block_4_expand_BN/gamma/Initializer/ones**
_class 
loc:@block_4_expand_BN/gamma*
dtype0
�
+block_4_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_4_expand_BN/gamma**
_class 
loc:@block_4_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
(block_4_expand_BN/beta/Initializer/zerosConst*
valueB�*    *)
_class
loc:@block_4_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_4_expand_BN/betaVarHandleOp*
shape:�*'
shared_nameblock_4_expand_BN/beta*)
_class
loc:@block_4_expand_BN/beta*
dtype0*
_output_shapes
: 
}
7block_4_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_4_expand_BN/beta*
_output_shapes
: 
�
block_4_expand_BN/beta/AssignAssignVariableOpblock_4_expand_BN/beta(block_4_expand_BN/beta/Initializer/zeros*)
_class
loc:@block_4_expand_BN/beta*
dtype0
�
*block_4_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_4_expand_BN/beta*)
_class
loc:@block_4_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_4_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *0
_class&
$"loc:@block_4_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_4_expand_BN/moving_meanVarHandleOp*
shape:�*.
shared_nameblock_4_expand_BN/moving_mean*0
_class&
$"loc:@block_4_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
>block_4_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_4_expand_BN/moving_mean*
_output_shapes
: 
�
$block_4_expand_BN/moving_mean/AssignAssignVariableOpblock_4_expand_BN/moving_mean/block_4_expand_BN/moving_mean/Initializer/zeros*0
_class&
$"loc:@block_4_expand_BN/moving_mean*
dtype0
�
1block_4_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_4_expand_BN/moving_mean*0
_class&
$"loc:@block_4_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_4_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*4
_class*
(&loc:@block_4_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_4_expand_BN/moving_varianceVarHandleOp*
shape:�*2
shared_name#!block_4_expand_BN/moving_variance*4
_class*
(&loc:@block_4_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Bblock_4_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_4_expand_BN/moving_variance*
_output_shapes
: 
�
(block_4_expand_BN/moving_variance/AssignAssignVariableOp!block_4_expand_BN/moving_variance2block_4_expand_BN/moving_variance/Initializer/ones*4
_class*
(&loc:@block_4_expand_BN/moving_variance*
dtype0
�
5block_4_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp!block_4_expand_BN/moving_variance*4
_class*
(&loc:@block_4_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
u
 block_4_expand_BN/ReadVariableOpReadVariableOpblock_4_expand_BN/gamma*
dtype0*
_output_shapes	
:�
v
"block_4_expand_BN/ReadVariableOp_1ReadVariableOpblock_4_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_4_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_4_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
1block_4_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp!block_4_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
 block_4_expand_BN/FusedBatchNormFusedBatchNormblock_4_expand/Conv2D block_4_expand_BN/ReadVariableOp"block_4_expand_BN/ReadVariableOp_1/block_4_expand_BN/FusedBatchNorm/ReadVariableOp1block_4_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
\
block_4_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 

block_4_expand_relu/Relu6Relu6 block_4_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
Cblock_4_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      �      *5
_class+
)'loc:@block_4_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Ablock_4_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *��p�*5
_class+
)'loc:@block_4_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Ablock_4_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *��p=*5
_class+
)'loc:@block_4_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Kblock_4_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformCblock_4_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*5
_class+
)'loc:@block_4_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Ablock_4_depthwise/depthwise_kernel/Initializer/random_uniform/subSubAblock_4_depthwise/depthwise_kernel/Initializer/random_uniform/maxAblock_4_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_4_depthwise/depthwise_kernel*
_output_shapes
: 
�
Ablock_4_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulKblock_4_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformAblock_4_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@block_4_depthwise/depthwise_kernel*'
_output_shapes
:�
�
=block_4_depthwise/depthwise_kernel/Initializer/random_uniformAddAblock_4_depthwise/depthwise_kernel/Initializer/random_uniform/mulAblock_4_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_4_depthwise/depthwise_kernel*'
_output_shapes
:�
�
"block_4_depthwise/depthwise_kernelVarHandleOp*
shape:�*3
shared_name$"block_4_depthwise/depthwise_kernel*5
_class+
)'loc:@block_4_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Cblock_4_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_4_depthwise/depthwise_kernel*
_output_shapes
: 
�
)block_4_depthwise/depthwise_kernel/AssignAssignVariableOp"block_4_depthwise/depthwise_kernel=block_4_depthwise/depthwise_kernel/Initializer/random_uniform*5
_class+
)'loc:@block_4_depthwise/depthwise_kernel*
dtype0
�
6block_4_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp"block_4_depthwise/depthwise_kernel*5
_class+
)'loc:@block_4_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
*block_4_depthwise/depthwise/ReadVariableOpReadVariableOp"block_4_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
z
!block_4_depthwise/depthwise/ShapeConst*%
valueB"      �      *
dtype0*
_output_shapes
:
z
)block_4_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_4_depthwise/depthwiseDepthwiseConv2dNativeblock_4_expand_relu/Relu6*block_4_depthwise/depthwise/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
+block_4_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*-
_class#
!loc:@block_4_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_4_depthwise_BN/gammaVarHandleOp*
shape:�*+
shared_nameblock_4_depthwise_BN/gamma*-
_class#
!loc:@block_4_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
;block_4_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_4_depthwise_BN/gamma*
_output_shapes
: 
�
!block_4_depthwise_BN/gamma/AssignAssignVariableOpblock_4_depthwise_BN/gamma+block_4_depthwise_BN/gamma/Initializer/ones*-
_class#
!loc:@block_4_depthwise_BN/gamma*
dtype0
�
.block_4_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_4_depthwise_BN/gamma*-
_class#
!loc:@block_4_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
+block_4_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *,
_class"
 loc:@block_4_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_4_depthwise_BN/betaVarHandleOp*
shape:�**
shared_nameblock_4_depthwise_BN/beta*,
_class"
 loc:@block_4_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
:block_4_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_4_depthwise_BN/beta*
_output_shapes
: 
�
 block_4_depthwise_BN/beta/AssignAssignVariableOpblock_4_depthwise_BN/beta+block_4_depthwise_BN/beta/Initializer/zeros*,
_class"
 loc:@block_4_depthwise_BN/beta*
dtype0
�
-block_4_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_4_depthwise_BN/beta*,
_class"
 loc:@block_4_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_4_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *3
_class)
'%loc:@block_4_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
 block_4_depthwise_BN/moving_meanVarHandleOp*
shape:�*1
shared_name" block_4_depthwise_BN/moving_mean*3
_class)
'%loc:@block_4_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Ablock_4_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp block_4_depthwise_BN/moving_mean*
_output_shapes
: 
�
'block_4_depthwise_BN/moving_mean/AssignAssignVariableOp block_4_depthwise_BN/moving_mean2block_4_depthwise_BN/moving_mean/Initializer/zeros*3
_class)
'%loc:@block_4_depthwise_BN/moving_mean*
dtype0
�
4block_4_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp block_4_depthwise_BN/moving_mean*3
_class)
'%loc:@block_4_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_4_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*7
_class-
+)loc:@block_4_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_4_depthwise_BN/moving_varianceVarHandleOp*
shape:�*5
shared_name&$block_4_depthwise_BN/moving_variance*7
_class-
+)loc:@block_4_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Eblock_4_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp$block_4_depthwise_BN/moving_variance*
_output_shapes
: 
�
+block_4_depthwise_BN/moving_variance/AssignAssignVariableOp$block_4_depthwise_BN/moving_variance5block_4_depthwise_BN/moving_variance/Initializer/ones*7
_class-
+)loc:@block_4_depthwise_BN/moving_variance*
dtype0
�
8block_4_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp$block_4_depthwise_BN/moving_variance*7
_class-
+)loc:@block_4_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
{
#block_4_depthwise_BN/ReadVariableOpReadVariableOpblock_4_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
|
%block_4_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_4_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_4_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp block_4_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
4block_4_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp$block_4_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
#block_4_depthwise_BN/FusedBatchNormFusedBatchNormblock_4_depthwise/depthwise#block_4_depthwise_BN/ReadVariableOp%block_4_depthwise_BN/ReadVariableOp_12block_4_depthwise_BN/FusedBatchNorm/ReadVariableOp4block_4_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
_
block_4_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_4_depthwise_relu/Relu6Relu6#block_4_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
7block_4_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �       *)
_class
loc:@block_4_project/kernel*
dtype0*
_output_shapes
:
�
5block_4_project/kernel/Initializer/random_uniform/minConst*
valueB
 *b�'�*)
_class
loc:@block_4_project/kernel*
dtype0*
_output_shapes
: 
�
5block_4_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *b�'>*)
_class
loc:@block_4_project/kernel*
dtype0*
_output_shapes
: 
�
?block_4_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_4_project/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_4_project/kernel*
dtype0*'
_output_shapes
:� 
�
5block_4_project/kernel/Initializer/random_uniform/subSub5block_4_project/kernel/Initializer/random_uniform/max5block_4_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_4_project/kernel*
_output_shapes
: 
�
5block_4_project/kernel/Initializer/random_uniform/mulMul?block_4_project/kernel/Initializer/random_uniform/RandomUniform5block_4_project/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_4_project/kernel*'
_output_shapes
:� 
�
1block_4_project/kernel/Initializer/random_uniformAdd5block_4_project/kernel/Initializer/random_uniform/mul5block_4_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_4_project/kernel*'
_output_shapes
:� 
�
block_4_project/kernelVarHandleOp*
shape:� *'
shared_nameblock_4_project/kernel*)
_class
loc:@block_4_project/kernel*
dtype0*
_output_shapes
: 
}
7block_4_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_4_project/kernel*
_output_shapes
: 
�
block_4_project/kernel/AssignAssignVariableOpblock_4_project/kernel1block_4_project/kernel/Initializer/random_uniform*)
_class
loc:@block_4_project/kernel*
dtype0
�
*block_4_project/kernel/Read/ReadVariableOpReadVariableOpblock_4_project/kernel*)
_class
loc:@block_4_project/kernel*
dtype0*'
_output_shapes
:� 
n
block_4_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_4_project/Conv2D/ReadVariableOpReadVariableOpblock_4_project/kernel*
dtype0*'
_output_shapes
:� 
�
block_4_project/Conv2DConv2Dblock_4_depthwise_relu/Relu6%block_4_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:��������� 
�
)block_4_project_BN/gamma/Initializer/onesConst*
valueB *  �?*+
_class!
loc:@block_4_project_BN/gamma*
dtype0*
_output_shapes
: 
�
block_4_project_BN/gammaVarHandleOp*
shape: *)
shared_nameblock_4_project_BN/gamma*+
_class!
loc:@block_4_project_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_4_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_4_project_BN/gamma*
_output_shapes
: 
�
block_4_project_BN/gamma/AssignAssignVariableOpblock_4_project_BN/gamma)block_4_project_BN/gamma/Initializer/ones*+
_class!
loc:@block_4_project_BN/gamma*
dtype0
�
,block_4_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_4_project_BN/gamma*+
_class!
loc:@block_4_project_BN/gamma*
dtype0*
_output_shapes
: 
�
)block_4_project_BN/beta/Initializer/zerosConst*
valueB *    **
_class 
loc:@block_4_project_BN/beta*
dtype0*
_output_shapes
: 
�
block_4_project_BN/betaVarHandleOp*
shape: *(
shared_nameblock_4_project_BN/beta**
_class 
loc:@block_4_project_BN/beta*
dtype0*
_output_shapes
: 

8block_4_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_4_project_BN/beta*
_output_shapes
: 
�
block_4_project_BN/beta/AssignAssignVariableOpblock_4_project_BN/beta)block_4_project_BN/beta/Initializer/zeros**
_class 
loc:@block_4_project_BN/beta*
dtype0
�
+block_4_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_4_project_BN/beta**
_class 
loc:@block_4_project_BN/beta*
dtype0*
_output_shapes
: 
�
0block_4_project_BN/moving_mean/Initializer/zerosConst*
valueB *    *1
_class'
%#loc:@block_4_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
block_4_project_BN/moving_meanVarHandleOp*
shape: */
shared_name block_4_project_BN/moving_mean*1
_class'
%#loc:@block_4_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_4_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_4_project_BN/moving_mean*
_output_shapes
: 
�
%block_4_project_BN/moving_mean/AssignAssignVariableOpblock_4_project_BN/moving_mean0block_4_project_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_4_project_BN/moving_mean*
dtype0
�
2block_4_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_4_project_BN/moving_mean*1
_class'
%#loc:@block_4_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
3block_4_project_BN/moving_variance/Initializer/onesConst*
valueB *  �?*5
_class+
)'loc:@block_4_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
"block_4_project_BN/moving_varianceVarHandleOp*
shape: *3
shared_name$"block_4_project_BN/moving_variance*5
_class+
)'loc:@block_4_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_4_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_4_project_BN/moving_variance*
_output_shapes
: 
�
)block_4_project_BN/moving_variance/AssignAssignVariableOp"block_4_project_BN/moving_variance3block_4_project_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_4_project_BN/moving_variance*
dtype0
�
6block_4_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_4_project_BN/moving_variance*5
_class+
)'loc:@block_4_project_BN/moving_variance*
dtype0*
_output_shapes
: 
v
!block_4_project_BN/ReadVariableOpReadVariableOpblock_4_project_BN/gamma*
dtype0*
_output_shapes
: 
w
#block_4_project_BN/ReadVariableOp_1ReadVariableOpblock_4_project_BN/beta*
dtype0*
_output_shapes
: 
�
0block_4_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_4_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
2block_4_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_4_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
!block_4_project_BN/FusedBatchNormFusedBatchNormblock_4_project/Conv2D!block_4_project_BN/ReadVariableOp#block_4_project_BN/ReadVariableOp_10block_4_project_BN/FusedBatchNorm/ReadVariableOp2block_4_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:��������� : : : : *
is_training( 
]
block_4_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_4_add/addAdd!block_3_project_BN/FusedBatchNorm!block_4_project_BN/FusedBatchNorm*
T0*/
_output_shapes
:��������� 
�
6block_5_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"          �   *(
_class
loc:@block_5_expand/kernel*
dtype0*
_output_shapes
:
�
4block_5_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *b�'�*(
_class
loc:@block_5_expand/kernel*
dtype0*
_output_shapes
: 
�
4block_5_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *b�'>*(
_class
loc:@block_5_expand/kernel*
dtype0*
_output_shapes
: 
�
>block_5_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform6block_5_expand/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@block_5_expand/kernel*
dtype0*'
_output_shapes
: �
�
4block_5_expand/kernel/Initializer/random_uniform/subSub4block_5_expand/kernel/Initializer/random_uniform/max4block_5_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_5_expand/kernel*
_output_shapes
: 
�
4block_5_expand/kernel/Initializer/random_uniform/mulMul>block_5_expand/kernel/Initializer/random_uniform/RandomUniform4block_5_expand/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@block_5_expand/kernel*'
_output_shapes
: �
�
0block_5_expand/kernel/Initializer/random_uniformAdd4block_5_expand/kernel/Initializer/random_uniform/mul4block_5_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_5_expand/kernel*'
_output_shapes
: �
�
block_5_expand/kernelVarHandleOp*
shape: �*&
shared_nameblock_5_expand/kernel*(
_class
loc:@block_5_expand/kernel*
dtype0*
_output_shapes
: 
{
6block_5_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_5_expand/kernel*
_output_shapes
: 
�
block_5_expand/kernel/AssignAssignVariableOpblock_5_expand/kernel0block_5_expand/kernel/Initializer/random_uniform*(
_class
loc:@block_5_expand/kernel*
dtype0
�
)block_5_expand/kernel/Read/ReadVariableOpReadVariableOpblock_5_expand/kernel*(
_class
loc:@block_5_expand/kernel*
dtype0*'
_output_shapes
: �
m
block_5_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
$block_5_expand/Conv2D/ReadVariableOpReadVariableOpblock_5_expand/kernel*
dtype0*'
_output_shapes
: �
�
block_5_expand/Conv2DConv2Dblock_4_add/add$block_5_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
(block_5_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?**
_class 
loc:@block_5_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_5_expand_BN/gammaVarHandleOp*
shape:�*(
shared_nameblock_5_expand_BN/gamma**
_class 
loc:@block_5_expand_BN/gamma*
dtype0*
_output_shapes
: 

8block_5_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_5_expand_BN/gamma*
_output_shapes
: 
�
block_5_expand_BN/gamma/AssignAssignVariableOpblock_5_expand_BN/gamma(block_5_expand_BN/gamma/Initializer/ones**
_class 
loc:@block_5_expand_BN/gamma*
dtype0
�
+block_5_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_5_expand_BN/gamma**
_class 
loc:@block_5_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
(block_5_expand_BN/beta/Initializer/zerosConst*
valueB�*    *)
_class
loc:@block_5_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_5_expand_BN/betaVarHandleOp*
shape:�*'
shared_nameblock_5_expand_BN/beta*)
_class
loc:@block_5_expand_BN/beta*
dtype0*
_output_shapes
: 
}
7block_5_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_5_expand_BN/beta*
_output_shapes
: 
�
block_5_expand_BN/beta/AssignAssignVariableOpblock_5_expand_BN/beta(block_5_expand_BN/beta/Initializer/zeros*)
_class
loc:@block_5_expand_BN/beta*
dtype0
�
*block_5_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_5_expand_BN/beta*)
_class
loc:@block_5_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_5_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *0
_class&
$"loc:@block_5_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_5_expand_BN/moving_meanVarHandleOp*
shape:�*.
shared_nameblock_5_expand_BN/moving_mean*0
_class&
$"loc:@block_5_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
>block_5_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_5_expand_BN/moving_mean*
_output_shapes
: 
�
$block_5_expand_BN/moving_mean/AssignAssignVariableOpblock_5_expand_BN/moving_mean/block_5_expand_BN/moving_mean/Initializer/zeros*0
_class&
$"loc:@block_5_expand_BN/moving_mean*
dtype0
�
1block_5_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_5_expand_BN/moving_mean*0
_class&
$"loc:@block_5_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_5_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*4
_class*
(&loc:@block_5_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_5_expand_BN/moving_varianceVarHandleOp*
shape:�*2
shared_name#!block_5_expand_BN/moving_variance*4
_class*
(&loc:@block_5_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Bblock_5_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_5_expand_BN/moving_variance*
_output_shapes
: 
�
(block_5_expand_BN/moving_variance/AssignAssignVariableOp!block_5_expand_BN/moving_variance2block_5_expand_BN/moving_variance/Initializer/ones*4
_class*
(&loc:@block_5_expand_BN/moving_variance*
dtype0
�
5block_5_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp!block_5_expand_BN/moving_variance*4
_class*
(&loc:@block_5_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
u
 block_5_expand_BN/ReadVariableOpReadVariableOpblock_5_expand_BN/gamma*
dtype0*
_output_shapes	
:�
v
"block_5_expand_BN/ReadVariableOp_1ReadVariableOpblock_5_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_5_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_5_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
1block_5_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp!block_5_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
 block_5_expand_BN/FusedBatchNormFusedBatchNormblock_5_expand/Conv2D block_5_expand_BN/ReadVariableOp"block_5_expand_BN/ReadVariableOp_1/block_5_expand_BN/FusedBatchNorm/ReadVariableOp1block_5_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
\
block_5_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 

block_5_expand_relu/Relu6Relu6 block_5_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
Cblock_5_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      �      *5
_class+
)'loc:@block_5_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Ablock_5_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *��p�*5
_class+
)'loc:@block_5_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Ablock_5_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *��p=*5
_class+
)'loc:@block_5_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Kblock_5_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformCblock_5_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*5
_class+
)'loc:@block_5_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Ablock_5_depthwise/depthwise_kernel/Initializer/random_uniform/subSubAblock_5_depthwise/depthwise_kernel/Initializer/random_uniform/maxAblock_5_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_5_depthwise/depthwise_kernel*
_output_shapes
: 
�
Ablock_5_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulKblock_5_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformAblock_5_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@block_5_depthwise/depthwise_kernel*'
_output_shapes
:�
�
=block_5_depthwise/depthwise_kernel/Initializer/random_uniformAddAblock_5_depthwise/depthwise_kernel/Initializer/random_uniform/mulAblock_5_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_5_depthwise/depthwise_kernel*'
_output_shapes
:�
�
"block_5_depthwise/depthwise_kernelVarHandleOp*
shape:�*3
shared_name$"block_5_depthwise/depthwise_kernel*5
_class+
)'loc:@block_5_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Cblock_5_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_5_depthwise/depthwise_kernel*
_output_shapes
: 
�
)block_5_depthwise/depthwise_kernel/AssignAssignVariableOp"block_5_depthwise/depthwise_kernel=block_5_depthwise/depthwise_kernel/Initializer/random_uniform*5
_class+
)'loc:@block_5_depthwise/depthwise_kernel*
dtype0
�
6block_5_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp"block_5_depthwise/depthwise_kernel*5
_class+
)'loc:@block_5_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
*block_5_depthwise/depthwise/ReadVariableOpReadVariableOp"block_5_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
z
!block_5_depthwise/depthwise/ShapeConst*%
valueB"      �      *
dtype0*
_output_shapes
:
z
)block_5_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_5_depthwise/depthwiseDepthwiseConv2dNativeblock_5_expand_relu/Relu6*block_5_depthwise/depthwise/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
+block_5_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*-
_class#
!loc:@block_5_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_5_depthwise_BN/gammaVarHandleOp*
shape:�*+
shared_nameblock_5_depthwise_BN/gamma*-
_class#
!loc:@block_5_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
;block_5_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_5_depthwise_BN/gamma*
_output_shapes
: 
�
!block_5_depthwise_BN/gamma/AssignAssignVariableOpblock_5_depthwise_BN/gamma+block_5_depthwise_BN/gamma/Initializer/ones*-
_class#
!loc:@block_5_depthwise_BN/gamma*
dtype0
�
.block_5_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_5_depthwise_BN/gamma*-
_class#
!loc:@block_5_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
+block_5_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *,
_class"
 loc:@block_5_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_5_depthwise_BN/betaVarHandleOp*
shape:�**
shared_nameblock_5_depthwise_BN/beta*,
_class"
 loc:@block_5_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
:block_5_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_5_depthwise_BN/beta*
_output_shapes
: 
�
 block_5_depthwise_BN/beta/AssignAssignVariableOpblock_5_depthwise_BN/beta+block_5_depthwise_BN/beta/Initializer/zeros*,
_class"
 loc:@block_5_depthwise_BN/beta*
dtype0
�
-block_5_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_5_depthwise_BN/beta*,
_class"
 loc:@block_5_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_5_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *3
_class)
'%loc:@block_5_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
 block_5_depthwise_BN/moving_meanVarHandleOp*
shape:�*1
shared_name" block_5_depthwise_BN/moving_mean*3
_class)
'%loc:@block_5_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Ablock_5_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp block_5_depthwise_BN/moving_mean*
_output_shapes
: 
�
'block_5_depthwise_BN/moving_mean/AssignAssignVariableOp block_5_depthwise_BN/moving_mean2block_5_depthwise_BN/moving_mean/Initializer/zeros*3
_class)
'%loc:@block_5_depthwise_BN/moving_mean*
dtype0
�
4block_5_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp block_5_depthwise_BN/moving_mean*3
_class)
'%loc:@block_5_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_5_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*7
_class-
+)loc:@block_5_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_5_depthwise_BN/moving_varianceVarHandleOp*
shape:�*5
shared_name&$block_5_depthwise_BN/moving_variance*7
_class-
+)loc:@block_5_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Eblock_5_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp$block_5_depthwise_BN/moving_variance*
_output_shapes
: 
�
+block_5_depthwise_BN/moving_variance/AssignAssignVariableOp$block_5_depthwise_BN/moving_variance5block_5_depthwise_BN/moving_variance/Initializer/ones*7
_class-
+)loc:@block_5_depthwise_BN/moving_variance*
dtype0
�
8block_5_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp$block_5_depthwise_BN/moving_variance*7
_class-
+)loc:@block_5_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
{
#block_5_depthwise_BN/ReadVariableOpReadVariableOpblock_5_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
|
%block_5_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_5_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_5_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp block_5_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
4block_5_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp$block_5_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
#block_5_depthwise_BN/FusedBatchNormFusedBatchNormblock_5_depthwise/depthwise#block_5_depthwise_BN/ReadVariableOp%block_5_depthwise_BN/ReadVariableOp_12block_5_depthwise_BN/FusedBatchNorm/ReadVariableOp4block_5_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
_
block_5_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_5_depthwise_relu/Relu6Relu6#block_5_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
7block_5_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �       *)
_class
loc:@block_5_project/kernel*
dtype0*
_output_shapes
:
�
5block_5_project/kernel/Initializer/random_uniform/minConst*
valueB
 *b�'�*)
_class
loc:@block_5_project/kernel*
dtype0*
_output_shapes
: 
�
5block_5_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *b�'>*)
_class
loc:@block_5_project/kernel*
dtype0*
_output_shapes
: 
�
?block_5_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_5_project/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_5_project/kernel*
dtype0*'
_output_shapes
:� 
�
5block_5_project/kernel/Initializer/random_uniform/subSub5block_5_project/kernel/Initializer/random_uniform/max5block_5_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_5_project/kernel*
_output_shapes
: 
�
5block_5_project/kernel/Initializer/random_uniform/mulMul?block_5_project/kernel/Initializer/random_uniform/RandomUniform5block_5_project/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_5_project/kernel*'
_output_shapes
:� 
�
1block_5_project/kernel/Initializer/random_uniformAdd5block_5_project/kernel/Initializer/random_uniform/mul5block_5_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_5_project/kernel*'
_output_shapes
:� 
�
block_5_project/kernelVarHandleOp*
shape:� *'
shared_nameblock_5_project/kernel*)
_class
loc:@block_5_project/kernel*
dtype0*
_output_shapes
: 
}
7block_5_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_5_project/kernel*
_output_shapes
: 
�
block_5_project/kernel/AssignAssignVariableOpblock_5_project/kernel1block_5_project/kernel/Initializer/random_uniform*)
_class
loc:@block_5_project/kernel*
dtype0
�
*block_5_project/kernel/Read/ReadVariableOpReadVariableOpblock_5_project/kernel*)
_class
loc:@block_5_project/kernel*
dtype0*'
_output_shapes
:� 
n
block_5_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_5_project/Conv2D/ReadVariableOpReadVariableOpblock_5_project/kernel*
dtype0*'
_output_shapes
:� 
�
block_5_project/Conv2DConv2Dblock_5_depthwise_relu/Relu6%block_5_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:��������� 
�
)block_5_project_BN/gamma/Initializer/onesConst*
valueB *  �?*+
_class!
loc:@block_5_project_BN/gamma*
dtype0*
_output_shapes
: 
�
block_5_project_BN/gammaVarHandleOp*
shape: *)
shared_nameblock_5_project_BN/gamma*+
_class!
loc:@block_5_project_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_5_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_5_project_BN/gamma*
_output_shapes
: 
�
block_5_project_BN/gamma/AssignAssignVariableOpblock_5_project_BN/gamma)block_5_project_BN/gamma/Initializer/ones*+
_class!
loc:@block_5_project_BN/gamma*
dtype0
�
,block_5_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_5_project_BN/gamma*+
_class!
loc:@block_5_project_BN/gamma*
dtype0*
_output_shapes
: 
�
)block_5_project_BN/beta/Initializer/zerosConst*
valueB *    **
_class 
loc:@block_5_project_BN/beta*
dtype0*
_output_shapes
: 
�
block_5_project_BN/betaVarHandleOp*
shape: *(
shared_nameblock_5_project_BN/beta**
_class 
loc:@block_5_project_BN/beta*
dtype0*
_output_shapes
: 

8block_5_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_5_project_BN/beta*
_output_shapes
: 
�
block_5_project_BN/beta/AssignAssignVariableOpblock_5_project_BN/beta)block_5_project_BN/beta/Initializer/zeros**
_class 
loc:@block_5_project_BN/beta*
dtype0
�
+block_5_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_5_project_BN/beta**
_class 
loc:@block_5_project_BN/beta*
dtype0*
_output_shapes
: 
�
0block_5_project_BN/moving_mean/Initializer/zerosConst*
valueB *    *1
_class'
%#loc:@block_5_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
block_5_project_BN/moving_meanVarHandleOp*
shape: */
shared_name block_5_project_BN/moving_mean*1
_class'
%#loc:@block_5_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_5_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_5_project_BN/moving_mean*
_output_shapes
: 
�
%block_5_project_BN/moving_mean/AssignAssignVariableOpblock_5_project_BN/moving_mean0block_5_project_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_5_project_BN/moving_mean*
dtype0
�
2block_5_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_5_project_BN/moving_mean*1
_class'
%#loc:@block_5_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
3block_5_project_BN/moving_variance/Initializer/onesConst*
valueB *  �?*5
_class+
)'loc:@block_5_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
"block_5_project_BN/moving_varianceVarHandleOp*
shape: *3
shared_name$"block_5_project_BN/moving_variance*5
_class+
)'loc:@block_5_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_5_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_5_project_BN/moving_variance*
_output_shapes
: 
�
)block_5_project_BN/moving_variance/AssignAssignVariableOp"block_5_project_BN/moving_variance3block_5_project_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_5_project_BN/moving_variance*
dtype0
�
6block_5_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_5_project_BN/moving_variance*5
_class+
)'loc:@block_5_project_BN/moving_variance*
dtype0*
_output_shapes
: 
v
!block_5_project_BN/ReadVariableOpReadVariableOpblock_5_project_BN/gamma*
dtype0*
_output_shapes
: 
w
#block_5_project_BN/ReadVariableOp_1ReadVariableOpblock_5_project_BN/beta*
dtype0*
_output_shapes
: 
�
0block_5_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_5_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
2block_5_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_5_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
!block_5_project_BN/FusedBatchNormFusedBatchNormblock_5_project/Conv2D!block_5_project_BN/ReadVariableOp#block_5_project_BN/ReadVariableOp_10block_5_project_BN/FusedBatchNorm/ReadVariableOp2block_5_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:��������� : : : : *
is_training( 
]
block_5_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_5_add/addAddblock_4_add/add!block_5_project_BN/FusedBatchNorm*
T0*/
_output_shapes
:��������� 
�
6block_6_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"          �   *(
_class
loc:@block_6_expand/kernel*
dtype0*
_output_shapes
:
�
4block_6_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *b�'�*(
_class
loc:@block_6_expand/kernel*
dtype0*
_output_shapes
: 
�
4block_6_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *b�'>*(
_class
loc:@block_6_expand/kernel*
dtype0*
_output_shapes
: 
�
>block_6_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform6block_6_expand/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@block_6_expand/kernel*
dtype0*'
_output_shapes
: �
�
4block_6_expand/kernel/Initializer/random_uniform/subSub4block_6_expand/kernel/Initializer/random_uniform/max4block_6_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_6_expand/kernel*
_output_shapes
: 
�
4block_6_expand/kernel/Initializer/random_uniform/mulMul>block_6_expand/kernel/Initializer/random_uniform/RandomUniform4block_6_expand/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@block_6_expand/kernel*'
_output_shapes
: �
�
0block_6_expand/kernel/Initializer/random_uniformAdd4block_6_expand/kernel/Initializer/random_uniform/mul4block_6_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_6_expand/kernel*'
_output_shapes
: �
�
block_6_expand/kernelVarHandleOp*
shape: �*&
shared_nameblock_6_expand/kernel*(
_class
loc:@block_6_expand/kernel*
dtype0*
_output_shapes
: 
{
6block_6_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_6_expand/kernel*
_output_shapes
: 
�
block_6_expand/kernel/AssignAssignVariableOpblock_6_expand/kernel0block_6_expand/kernel/Initializer/random_uniform*(
_class
loc:@block_6_expand/kernel*
dtype0
�
)block_6_expand/kernel/Read/ReadVariableOpReadVariableOpblock_6_expand/kernel*(
_class
loc:@block_6_expand/kernel*
dtype0*'
_output_shapes
: �
m
block_6_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
$block_6_expand/Conv2D/ReadVariableOpReadVariableOpblock_6_expand/kernel*
dtype0*'
_output_shapes
: �
�
block_6_expand/Conv2DConv2Dblock_5_add/add$block_6_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
(block_6_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?**
_class 
loc:@block_6_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_6_expand_BN/gammaVarHandleOp*
shape:�*(
shared_nameblock_6_expand_BN/gamma**
_class 
loc:@block_6_expand_BN/gamma*
dtype0*
_output_shapes
: 

8block_6_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_6_expand_BN/gamma*
_output_shapes
: 
�
block_6_expand_BN/gamma/AssignAssignVariableOpblock_6_expand_BN/gamma(block_6_expand_BN/gamma/Initializer/ones**
_class 
loc:@block_6_expand_BN/gamma*
dtype0
�
+block_6_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_6_expand_BN/gamma**
_class 
loc:@block_6_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
(block_6_expand_BN/beta/Initializer/zerosConst*
valueB�*    *)
_class
loc:@block_6_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_6_expand_BN/betaVarHandleOp*
shape:�*'
shared_nameblock_6_expand_BN/beta*)
_class
loc:@block_6_expand_BN/beta*
dtype0*
_output_shapes
: 
}
7block_6_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_6_expand_BN/beta*
_output_shapes
: 
�
block_6_expand_BN/beta/AssignAssignVariableOpblock_6_expand_BN/beta(block_6_expand_BN/beta/Initializer/zeros*)
_class
loc:@block_6_expand_BN/beta*
dtype0
�
*block_6_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_6_expand_BN/beta*)
_class
loc:@block_6_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_6_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *0
_class&
$"loc:@block_6_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_6_expand_BN/moving_meanVarHandleOp*
shape:�*.
shared_nameblock_6_expand_BN/moving_mean*0
_class&
$"loc:@block_6_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
>block_6_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_6_expand_BN/moving_mean*
_output_shapes
: 
�
$block_6_expand_BN/moving_mean/AssignAssignVariableOpblock_6_expand_BN/moving_mean/block_6_expand_BN/moving_mean/Initializer/zeros*0
_class&
$"loc:@block_6_expand_BN/moving_mean*
dtype0
�
1block_6_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_6_expand_BN/moving_mean*0
_class&
$"loc:@block_6_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_6_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*4
_class*
(&loc:@block_6_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_6_expand_BN/moving_varianceVarHandleOp*
shape:�*2
shared_name#!block_6_expand_BN/moving_variance*4
_class*
(&loc:@block_6_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Bblock_6_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_6_expand_BN/moving_variance*
_output_shapes
: 
�
(block_6_expand_BN/moving_variance/AssignAssignVariableOp!block_6_expand_BN/moving_variance2block_6_expand_BN/moving_variance/Initializer/ones*4
_class*
(&loc:@block_6_expand_BN/moving_variance*
dtype0
�
5block_6_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp!block_6_expand_BN/moving_variance*4
_class*
(&loc:@block_6_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
u
 block_6_expand_BN/ReadVariableOpReadVariableOpblock_6_expand_BN/gamma*
dtype0*
_output_shapes	
:�
v
"block_6_expand_BN/ReadVariableOp_1ReadVariableOpblock_6_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_6_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_6_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
1block_6_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp!block_6_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
 block_6_expand_BN/FusedBatchNormFusedBatchNormblock_6_expand/Conv2D block_6_expand_BN/ReadVariableOp"block_6_expand_BN/ReadVariableOp_1/block_6_expand_BN/FusedBatchNorm/ReadVariableOp1block_6_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
\
block_6_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 

block_6_expand_relu/Relu6Relu6 block_6_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
block_6_pad/Pad/paddingsConst*9
value0B."                               *
dtype0*
_output_shapes

:
�
block_6_pad/PadPadblock_6_expand_relu/Relu6block_6_pad/Pad/paddings*
T0*0
_output_shapes
:����������
�
Cblock_6_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      �      *5
_class+
)'loc:@block_6_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Ablock_6_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *��p�*5
_class+
)'loc:@block_6_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Ablock_6_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *��p=*5
_class+
)'loc:@block_6_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Kblock_6_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformCblock_6_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*5
_class+
)'loc:@block_6_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Ablock_6_depthwise/depthwise_kernel/Initializer/random_uniform/subSubAblock_6_depthwise/depthwise_kernel/Initializer/random_uniform/maxAblock_6_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_6_depthwise/depthwise_kernel*
_output_shapes
: 
�
Ablock_6_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulKblock_6_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformAblock_6_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@block_6_depthwise/depthwise_kernel*'
_output_shapes
:�
�
=block_6_depthwise/depthwise_kernel/Initializer/random_uniformAddAblock_6_depthwise/depthwise_kernel/Initializer/random_uniform/mulAblock_6_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_6_depthwise/depthwise_kernel*'
_output_shapes
:�
�
"block_6_depthwise/depthwise_kernelVarHandleOp*
shape:�*3
shared_name$"block_6_depthwise/depthwise_kernel*5
_class+
)'loc:@block_6_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Cblock_6_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_6_depthwise/depthwise_kernel*
_output_shapes
: 
�
)block_6_depthwise/depthwise_kernel/AssignAssignVariableOp"block_6_depthwise/depthwise_kernel=block_6_depthwise/depthwise_kernel/Initializer/random_uniform*5
_class+
)'loc:@block_6_depthwise/depthwise_kernel*
dtype0
�
6block_6_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp"block_6_depthwise/depthwise_kernel*5
_class+
)'loc:@block_6_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
*block_6_depthwise/depthwise/ReadVariableOpReadVariableOp"block_6_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
z
!block_6_depthwise/depthwise/ShapeConst*%
valueB"      �      *
dtype0*
_output_shapes
:
z
)block_6_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_6_depthwise/depthwiseDepthwiseConv2dNativeblock_6_pad/Pad*block_6_depthwise/depthwise/ReadVariableOp*
paddingVALID*
T0*
strides
*0
_output_shapes
:����������
�
+block_6_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*-
_class#
!loc:@block_6_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_6_depthwise_BN/gammaVarHandleOp*
shape:�*+
shared_nameblock_6_depthwise_BN/gamma*-
_class#
!loc:@block_6_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
;block_6_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_6_depthwise_BN/gamma*
_output_shapes
: 
�
!block_6_depthwise_BN/gamma/AssignAssignVariableOpblock_6_depthwise_BN/gamma+block_6_depthwise_BN/gamma/Initializer/ones*-
_class#
!loc:@block_6_depthwise_BN/gamma*
dtype0
�
.block_6_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_6_depthwise_BN/gamma*-
_class#
!loc:@block_6_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
+block_6_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *,
_class"
 loc:@block_6_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_6_depthwise_BN/betaVarHandleOp*
shape:�**
shared_nameblock_6_depthwise_BN/beta*,
_class"
 loc:@block_6_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
:block_6_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_6_depthwise_BN/beta*
_output_shapes
: 
�
 block_6_depthwise_BN/beta/AssignAssignVariableOpblock_6_depthwise_BN/beta+block_6_depthwise_BN/beta/Initializer/zeros*,
_class"
 loc:@block_6_depthwise_BN/beta*
dtype0
�
-block_6_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_6_depthwise_BN/beta*,
_class"
 loc:@block_6_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_6_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *3
_class)
'%loc:@block_6_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
 block_6_depthwise_BN/moving_meanVarHandleOp*
shape:�*1
shared_name" block_6_depthwise_BN/moving_mean*3
_class)
'%loc:@block_6_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Ablock_6_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp block_6_depthwise_BN/moving_mean*
_output_shapes
: 
�
'block_6_depthwise_BN/moving_mean/AssignAssignVariableOp block_6_depthwise_BN/moving_mean2block_6_depthwise_BN/moving_mean/Initializer/zeros*3
_class)
'%loc:@block_6_depthwise_BN/moving_mean*
dtype0
�
4block_6_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp block_6_depthwise_BN/moving_mean*3
_class)
'%loc:@block_6_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_6_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*7
_class-
+)loc:@block_6_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_6_depthwise_BN/moving_varianceVarHandleOp*
shape:�*5
shared_name&$block_6_depthwise_BN/moving_variance*7
_class-
+)loc:@block_6_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Eblock_6_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp$block_6_depthwise_BN/moving_variance*
_output_shapes
: 
�
+block_6_depthwise_BN/moving_variance/AssignAssignVariableOp$block_6_depthwise_BN/moving_variance5block_6_depthwise_BN/moving_variance/Initializer/ones*7
_class-
+)loc:@block_6_depthwise_BN/moving_variance*
dtype0
�
8block_6_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp$block_6_depthwise_BN/moving_variance*7
_class-
+)loc:@block_6_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
{
#block_6_depthwise_BN/ReadVariableOpReadVariableOpblock_6_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
|
%block_6_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_6_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_6_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp block_6_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
4block_6_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp$block_6_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
#block_6_depthwise_BN/FusedBatchNormFusedBatchNormblock_6_depthwise/depthwise#block_6_depthwise_BN/ReadVariableOp%block_6_depthwise_BN/ReadVariableOp_12block_6_depthwise_BN/FusedBatchNorm/ReadVariableOp4block_6_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
_
block_6_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_6_depthwise_relu/Relu6Relu6#block_6_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
7block_6_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �   @   *)
_class
loc:@block_6_project/kernel*
dtype0*
_output_shapes
:
�
5block_6_project/kernel/Initializer/random_uniform/minConst*
valueB
 *q��*)
_class
loc:@block_6_project/kernel*
dtype0*
_output_shapes
: 
�
5block_6_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *q�>*)
_class
loc:@block_6_project/kernel*
dtype0*
_output_shapes
: 
�
?block_6_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_6_project/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_6_project/kernel*
dtype0*'
_output_shapes
:�@
�
5block_6_project/kernel/Initializer/random_uniform/subSub5block_6_project/kernel/Initializer/random_uniform/max5block_6_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_6_project/kernel*
_output_shapes
: 
�
5block_6_project/kernel/Initializer/random_uniform/mulMul?block_6_project/kernel/Initializer/random_uniform/RandomUniform5block_6_project/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_6_project/kernel*'
_output_shapes
:�@
�
1block_6_project/kernel/Initializer/random_uniformAdd5block_6_project/kernel/Initializer/random_uniform/mul5block_6_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_6_project/kernel*'
_output_shapes
:�@
�
block_6_project/kernelVarHandleOp*
shape:�@*'
shared_nameblock_6_project/kernel*)
_class
loc:@block_6_project/kernel*
dtype0*
_output_shapes
: 
}
7block_6_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_6_project/kernel*
_output_shapes
: 
�
block_6_project/kernel/AssignAssignVariableOpblock_6_project/kernel1block_6_project/kernel/Initializer/random_uniform*)
_class
loc:@block_6_project/kernel*
dtype0
�
*block_6_project/kernel/Read/ReadVariableOpReadVariableOpblock_6_project/kernel*)
_class
loc:@block_6_project/kernel*
dtype0*'
_output_shapes
:�@
n
block_6_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_6_project/Conv2D/ReadVariableOpReadVariableOpblock_6_project/kernel*
dtype0*'
_output_shapes
:�@
�
block_6_project/Conv2DConv2Dblock_6_depthwise_relu/Relu6%block_6_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:���������@
�
)block_6_project_BN/gamma/Initializer/onesConst*
valueB@*  �?*+
_class!
loc:@block_6_project_BN/gamma*
dtype0*
_output_shapes
:@
�
block_6_project_BN/gammaVarHandleOp*
shape:@*)
shared_nameblock_6_project_BN/gamma*+
_class!
loc:@block_6_project_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_6_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_6_project_BN/gamma*
_output_shapes
: 
�
block_6_project_BN/gamma/AssignAssignVariableOpblock_6_project_BN/gamma)block_6_project_BN/gamma/Initializer/ones*+
_class!
loc:@block_6_project_BN/gamma*
dtype0
�
,block_6_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_6_project_BN/gamma*+
_class!
loc:@block_6_project_BN/gamma*
dtype0*
_output_shapes
:@
�
)block_6_project_BN/beta/Initializer/zerosConst*
valueB@*    **
_class 
loc:@block_6_project_BN/beta*
dtype0*
_output_shapes
:@
�
block_6_project_BN/betaVarHandleOp*
shape:@*(
shared_nameblock_6_project_BN/beta**
_class 
loc:@block_6_project_BN/beta*
dtype0*
_output_shapes
: 

8block_6_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_6_project_BN/beta*
_output_shapes
: 
�
block_6_project_BN/beta/AssignAssignVariableOpblock_6_project_BN/beta)block_6_project_BN/beta/Initializer/zeros**
_class 
loc:@block_6_project_BN/beta*
dtype0
�
+block_6_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_6_project_BN/beta**
_class 
loc:@block_6_project_BN/beta*
dtype0*
_output_shapes
:@
�
0block_6_project_BN/moving_mean/Initializer/zerosConst*
valueB@*    *1
_class'
%#loc:@block_6_project_BN/moving_mean*
dtype0*
_output_shapes
:@
�
block_6_project_BN/moving_meanVarHandleOp*
shape:@*/
shared_name block_6_project_BN/moving_mean*1
_class'
%#loc:@block_6_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_6_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_6_project_BN/moving_mean*
_output_shapes
: 
�
%block_6_project_BN/moving_mean/AssignAssignVariableOpblock_6_project_BN/moving_mean0block_6_project_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_6_project_BN/moving_mean*
dtype0
�
2block_6_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_6_project_BN/moving_mean*1
_class'
%#loc:@block_6_project_BN/moving_mean*
dtype0*
_output_shapes
:@
�
3block_6_project_BN/moving_variance/Initializer/onesConst*
valueB@*  �?*5
_class+
)'loc:@block_6_project_BN/moving_variance*
dtype0*
_output_shapes
:@
�
"block_6_project_BN/moving_varianceVarHandleOp*
shape:@*3
shared_name$"block_6_project_BN/moving_variance*5
_class+
)'loc:@block_6_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_6_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_6_project_BN/moving_variance*
_output_shapes
: 
�
)block_6_project_BN/moving_variance/AssignAssignVariableOp"block_6_project_BN/moving_variance3block_6_project_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_6_project_BN/moving_variance*
dtype0
�
6block_6_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_6_project_BN/moving_variance*5
_class+
)'loc:@block_6_project_BN/moving_variance*
dtype0*
_output_shapes
:@
v
!block_6_project_BN/ReadVariableOpReadVariableOpblock_6_project_BN/gamma*
dtype0*
_output_shapes
:@
w
#block_6_project_BN/ReadVariableOp_1ReadVariableOpblock_6_project_BN/beta*
dtype0*
_output_shapes
:@
�
0block_6_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_6_project_BN/moving_mean*
dtype0*
_output_shapes
:@
�
2block_6_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_6_project_BN/moving_variance*
dtype0*
_output_shapes
:@
�
!block_6_project_BN/FusedBatchNormFusedBatchNormblock_6_project/Conv2D!block_6_project_BN/ReadVariableOp#block_6_project_BN/ReadVariableOp_10block_6_project_BN/FusedBatchNorm/ReadVariableOp2block_6_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:���������@:@:@:@:@*
is_training( 
]
block_6_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
6block_7_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   �  *(
_class
loc:@block_7_expand/kernel*
dtype0*
_output_shapes
:
�
4block_7_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *���*(
_class
loc:@block_7_expand/kernel*
dtype0*
_output_shapes
: 
�
4block_7_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=*(
_class
loc:@block_7_expand/kernel*
dtype0*
_output_shapes
: 
�
>block_7_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform6block_7_expand/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@block_7_expand/kernel*
dtype0*'
_output_shapes
:@�
�
4block_7_expand/kernel/Initializer/random_uniform/subSub4block_7_expand/kernel/Initializer/random_uniform/max4block_7_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_7_expand/kernel*
_output_shapes
: 
�
4block_7_expand/kernel/Initializer/random_uniform/mulMul>block_7_expand/kernel/Initializer/random_uniform/RandomUniform4block_7_expand/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@block_7_expand/kernel*'
_output_shapes
:@�
�
0block_7_expand/kernel/Initializer/random_uniformAdd4block_7_expand/kernel/Initializer/random_uniform/mul4block_7_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_7_expand/kernel*'
_output_shapes
:@�
�
block_7_expand/kernelVarHandleOp*
shape:@�*&
shared_nameblock_7_expand/kernel*(
_class
loc:@block_7_expand/kernel*
dtype0*
_output_shapes
: 
{
6block_7_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_7_expand/kernel*
_output_shapes
: 
�
block_7_expand/kernel/AssignAssignVariableOpblock_7_expand/kernel0block_7_expand/kernel/Initializer/random_uniform*(
_class
loc:@block_7_expand/kernel*
dtype0
�
)block_7_expand/kernel/Read/ReadVariableOpReadVariableOpblock_7_expand/kernel*(
_class
loc:@block_7_expand/kernel*
dtype0*'
_output_shapes
:@�
m
block_7_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
$block_7_expand/Conv2D/ReadVariableOpReadVariableOpblock_7_expand/kernel*
dtype0*'
_output_shapes
:@�
�
block_7_expand/Conv2DConv2D!block_6_project_BN/FusedBatchNorm$block_7_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
(block_7_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?**
_class 
loc:@block_7_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_7_expand_BN/gammaVarHandleOp*
shape:�*(
shared_nameblock_7_expand_BN/gamma**
_class 
loc:@block_7_expand_BN/gamma*
dtype0*
_output_shapes
: 

8block_7_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_7_expand_BN/gamma*
_output_shapes
: 
�
block_7_expand_BN/gamma/AssignAssignVariableOpblock_7_expand_BN/gamma(block_7_expand_BN/gamma/Initializer/ones**
_class 
loc:@block_7_expand_BN/gamma*
dtype0
�
+block_7_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_7_expand_BN/gamma**
_class 
loc:@block_7_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
(block_7_expand_BN/beta/Initializer/zerosConst*
valueB�*    *)
_class
loc:@block_7_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_7_expand_BN/betaVarHandleOp*
shape:�*'
shared_nameblock_7_expand_BN/beta*)
_class
loc:@block_7_expand_BN/beta*
dtype0*
_output_shapes
: 
}
7block_7_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_7_expand_BN/beta*
_output_shapes
: 
�
block_7_expand_BN/beta/AssignAssignVariableOpblock_7_expand_BN/beta(block_7_expand_BN/beta/Initializer/zeros*)
_class
loc:@block_7_expand_BN/beta*
dtype0
�
*block_7_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_7_expand_BN/beta*)
_class
loc:@block_7_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_7_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *0
_class&
$"loc:@block_7_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_7_expand_BN/moving_meanVarHandleOp*
shape:�*.
shared_nameblock_7_expand_BN/moving_mean*0
_class&
$"loc:@block_7_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
>block_7_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_7_expand_BN/moving_mean*
_output_shapes
: 
�
$block_7_expand_BN/moving_mean/AssignAssignVariableOpblock_7_expand_BN/moving_mean/block_7_expand_BN/moving_mean/Initializer/zeros*0
_class&
$"loc:@block_7_expand_BN/moving_mean*
dtype0
�
1block_7_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_7_expand_BN/moving_mean*0
_class&
$"loc:@block_7_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_7_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*4
_class*
(&loc:@block_7_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_7_expand_BN/moving_varianceVarHandleOp*
shape:�*2
shared_name#!block_7_expand_BN/moving_variance*4
_class*
(&loc:@block_7_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Bblock_7_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_7_expand_BN/moving_variance*
_output_shapes
: 
�
(block_7_expand_BN/moving_variance/AssignAssignVariableOp!block_7_expand_BN/moving_variance2block_7_expand_BN/moving_variance/Initializer/ones*4
_class*
(&loc:@block_7_expand_BN/moving_variance*
dtype0
�
5block_7_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp!block_7_expand_BN/moving_variance*4
_class*
(&loc:@block_7_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
u
 block_7_expand_BN/ReadVariableOpReadVariableOpblock_7_expand_BN/gamma*
dtype0*
_output_shapes	
:�
v
"block_7_expand_BN/ReadVariableOp_1ReadVariableOpblock_7_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_7_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_7_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
1block_7_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp!block_7_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
 block_7_expand_BN/FusedBatchNormFusedBatchNormblock_7_expand/Conv2D block_7_expand_BN/ReadVariableOp"block_7_expand_BN/ReadVariableOp_1/block_7_expand_BN/FusedBatchNorm/ReadVariableOp1block_7_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
\
block_7_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 

block_7_expand_relu/Relu6Relu6 block_7_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
Cblock_7_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      �     *5
_class+
)'loc:@block_7_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Ablock_7_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *�q*�*5
_class+
)'loc:@block_7_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Ablock_7_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *�q*=*5
_class+
)'loc:@block_7_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Kblock_7_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformCblock_7_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*5
_class+
)'loc:@block_7_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Ablock_7_depthwise/depthwise_kernel/Initializer/random_uniform/subSubAblock_7_depthwise/depthwise_kernel/Initializer/random_uniform/maxAblock_7_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_7_depthwise/depthwise_kernel*
_output_shapes
: 
�
Ablock_7_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulKblock_7_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformAblock_7_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@block_7_depthwise/depthwise_kernel*'
_output_shapes
:�
�
=block_7_depthwise/depthwise_kernel/Initializer/random_uniformAddAblock_7_depthwise/depthwise_kernel/Initializer/random_uniform/mulAblock_7_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_7_depthwise/depthwise_kernel*'
_output_shapes
:�
�
"block_7_depthwise/depthwise_kernelVarHandleOp*
shape:�*3
shared_name$"block_7_depthwise/depthwise_kernel*5
_class+
)'loc:@block_7_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Cblock_7_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_7_depthwise/depthwise_kernel*
_output_shapes
: 
�
)block_7_depthwise/depthwise_kernel/AssignAssignVariableOp"block_7_depthwise/depthwise_kernel=block_7_depthwise/depthwise_kernel/Initializer/random_uniform*5
_class+
)'loc:@block_7_depthwise/depthwise_kernel*
dtype0
�
6block_7_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp"block_7_depthwise/depthwise_kernel*5
_class+
)'loc:@block_7_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
*block_7_depthwise/depthwise/ReadVariableOpReadVariableOp"block_7_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
z
!block_7_depthwise/depthwise/ShapeConst*%
valueB"      �     *
dtype0*
_output_shapes
:
z
)block_7_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_7_depthwise/depthwiseDepthwiseConv2dNativeblock_7_expand_relu/Relu6*block_7_depthwise/depthwise/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
+block_7_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*-
_class#
!loc:@block_7_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_7_depthwise_BN/gammaVarHandleOp*
shape:�*+
shared_nameblock_7_depthwise_BN/gamma*-
_class#
!loc:@block_7_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
;block_7_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_7_depthwise_BN/gamma*
_output_shapes
: 
�
!block_7_depthwise_BN/gamma/AssignAssignVariableOpblock_7_depthwise_BN/gamma+block_7_depthwise_BN/gamma/Initializer/ones*-
_class#
!loc:@block_7_depthwise_BN/gamma*
dtype0
�
.block_7_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_7_depthwise_BN/gamma*-
_class#
!loc:@block_7_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
+block_7_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *,
_class"
 loc:@block_7_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_7_depthwise_BN/betaVarHandleOp*
shape:�**
shared_nameblock_7_depthwise_BN/beta*,
_class"
 loc:@block_7_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
:block_7_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_7_depthwise_BN/beta*
_output_shapes
: 
�
 block_7_depthwise_BN/beta/AssignAssignVariableOpblock_7_depthwise_BN/beta+block_7_depthwise_BN/beta/Initializer/zeros*,
_class"
 loc:@block_7_depthwise_BN/beta*
dtype0
�
-block_7_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_7_depthwise_BN/beta*,
_class"
 loc:@block_7_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_7_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *3
_class)
'%loc:@block_7_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
 block_7_depthwise_BN/moving_meanVarHandleOp*
shape:�*1
shared_name" block_7_depthwise_BN/moving_mean*3
_class)
'%loc:@block_7_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Ablock_7_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp block_7_depthwise_BN/moving_mean*
_output_shapes
: 
�
'block_7_depthwise_BN/moving_mean/AssignAssignVariableOp block_7_depthwise_BN/moving_mean2block_7_depthwise_BN/moving_mean/Initializer/zeros*3
_class)
'%loc:@block_7_depthwise_BN/moving_mean*
dtype0
�
4block_7_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp block_7_depthwise_BN/moving_mean*3
_class)
'%loc:@block_7_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_7_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*7
_class-
+)loc:@block_7_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_7_depthwise_BN/moving_varianceVarHandleOp*
shape:�*5
shared_name&$block_7_depthwise_BN/moving_variance*7
_class-
+)loc:@block_7_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Eblock_7_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp$block_7_depthwise_BN/moving_variance*
_output_shapes
: 
�
+block_7_depthwise_BN/moving_variance/AssignAssignVariableOp$block_7_depthwise_BN/moving_variance5block_7_depthwise_BN/moving_variance/Initializer/ones*7
_class-
+)loc:@block_7_depthwise_BN/moving_variance*
dtype0
�
8block_7_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp$block_7_depthwise_BN/moving_variance*7
_class-
+)loc:@block_7_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
{
#block_7_depthwise_BN/ReadVariableOpReadVariableOpblock_7_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
|
%block_7_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_7_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_7_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp block_7_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
4block_7_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp$block_7_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
#block_7_depthwise_BN/FusedBatchNormFusedBatchNormblock_7_depthwise/depthwise#block_7_depthwise_BN/ReadVariableOp%block_7_depthwise_BN/ReadVariableOp_12block_7_depthwise_BN/FusedBatchNorm/ReadVariableOp4block_7_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
_
block_7_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_7_depthwise_relu/Relu6Relu6#block_7_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
7block_7_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �  @   *)
_class
loc:@block_7_project/kernel*
dtype0*
_output_shapes
:
�
5block_7_project/kernel/Initializer/random_uniform/minConst*
valueB
 *���*)
_class
loc:@block_7_project/kernel*
dtype0*
_output_shapes
: 
�
5block_7_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=*)
_class
loc:@block_7_project/kernel*
dtype0*
_output_shapes
: 
�
?block_7_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_7_project/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_7_project/kernel*
dtype0*'
_output_shapes
:�@
�
5block_7_project/kernel/Initializer/random_uniform/subSub5block_7_project/kernel/Initializer/random_uniform/max5block_7_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_7_project/kernel*
_output_shapes
: 
�
5block_7_project/kernel/Initializer/random_uniform/mulMul?block_7_project/kernel/Initializer/random_uniform/RandomUniform5block_7_project/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_7_project/kernel*'
_output_shapes
:�@
�
1block_7_project/kernel/Initializer/random_uniformAdd5block_7_project/kernel/Initializer/random_uniform/mul5block_7_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_7_project/kernel*'
_output_shapes
:�@
�
block_7_project/kernelVarHandleOp*
shape:�@*'
shared_nameblock_7_project/kernel*)
_class
loc:@block_7_project/kernel*
dtype0*
_output_shapes
: 
}
7block_7_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_7_project/kernel*
_output_shapes
: 
�
block_7_project/kernel/AssignAssignVariableOpblock_7_project/kernel1block_7_project/kernel/Initializer/random_uniform*)
_class
loc:@block_7_project/kernel*
dtype0
�
*block_7_project/kernel/Read/ReadVariableOpReadVariableOpblock_7_project/kernel*)
_class
loc:@block_7_project/kernel*
dtype0*'
_output_shapes
:�@
n
block_7_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_7_project/Conv2D/ReadVariableOpReadVariableOpblock_7_project/kernel*
dtype0*'
_output_shapes
:�@
�
block_7_project/Conv2DConv2Dblock_7_depthwise_relu/Relu6%block_7_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:���������@
�
)block_7_project_BN/gamma/Initializer/onesConst*
valueB@*  �?*+
_class!
loc:@block_7_project_BN/gamma*
dtype0*
_output_shapes
:@
�
block_7_project_BN/gammaVarHandleOp*
shape:@*)
shared_nameblock_7_project_BN/gamma*+
_class!
loc:@block_7_project_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_7_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_7_project_BN/gamma*
_output_shapes
: 
�
block_7_project_BN/gamma/AssignAssignVariableOpblock_7_project_BN/gamma)block_7_project_BN/gamma/Initializer/ones*+
_class!
loc:@block_7_project_BN/gamma*
dtype0
�
,block_7_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_7_project_BN/gamma*+
_class!
loc:@block_7_project_BN/gamma*
dtype0*
_output_shapes
:@
�
)block_7_project_BN/beta/Initializer/zerosConst*
valueB@*    **
_class 
loc:@block_7_project_BN/beta*
dtype0*
_output_shapes
:@
�
block_7_project_BN/betaVarHandleOp*
shape:@*(
shared_nameblock_7_project_BN/beta**
_class 
loc:@block_7_project_BN/beta*
dtype0*
_output_shapes
: 

8block_7_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_7_project_BN/beta*
_output_shapes
: 
�
block_7_project_BN/beta/AssignAssignVariableOpblock_7_project_BN/beta)block_7_project_BN/beta/Initializer/zeros**
_class 
loc:@block_7_project_BN/beta*
dtype0
�
+block_7_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_7_project_BN/beta**
_class 
loc:@block_7_project_BN/beta*
dtype0*
_output_shapes
:@
�
0block_7_project_BN/moving_mean/Initializer/zerosConst*
valueB@*    *1
_class'
%#loc:@block_7_project_BN/moving_mean*
dtype0*
_output_shapes
:@
�
block_7_project_BN/moving_meanVarHandleOp*
shape:@*/
shared_name block_7_project_BN/moving_mean*1
_class'
%#loc:@block_7_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_7_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_7_project_BN/moving_mean*
_output_shapes
: 
�
%block_7_project_BN/moving_mean/AssignAssignVariableOpblock_7_project_BN/moving_mean0block_7_project_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_7_project_BN/moving_mean*
dtype0
�
2block_7_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_7_project_BN/moving_mean*1
_class'
%#loc:@block_7_project_BN/moving_mean*
dtype0*
_output_shapes
:@
�
3block_7_project_BN/moving_variance/Initializer/onesConst*
valueB@*  �?*5
_class+
)'loc:@block_7_project_BN/moving_variance*
dtype0*
_output_shapes
:@
�
"block_7_project_BN/moving_varianceVarHandleOp*
shape:@*3
shared_name$"block_7_project_BN/moving_variance*5
_class+
)'loc:@block_7_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_7_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_7_project_BN/moving_variance*
_output_shapes
: 
�
)block_7_project_BN/moving_variance/AssignAssignVariableOp"block_7_project_BN/moving_variance3block_7_project_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_7_project_BN/moving_variance*
dtype0
�
6block_7_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_7_project_BN/moving_variance*5
_class+
)'loc:@block_7_project_BN/moving_variance*
dtype0*
_output_shapes
:@
v
!block_7_project_BN/ReadVariableOpReadVariableOpblock_7_project_BN/gamma*
dtype0*
_output_shapes
:@
w
#block_7_project_BN/ReadVariableOp_1ReadVariableOpblock_7_project_BN/beta*
dtype0*
_output_shapes
:@
�
0block_7_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_7_project_BN/moving_mean*
dtype0*
_output_shapes
:@
�
2block_7_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_7_project_BN/moving_variance*
dtype0*
_output_shapes
:@
�
!block_7_project_BN/FusedBatchNormFusedBatchNormblock_7_project/Conv2D!block_7_project_BN/ReadVariableOp#block_7_project_BN/ReadVariableOp_10block_7_project_BN/FusedBatchNorm/ReadVariableOp2block_7_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:���������@:@:@:@:@*
is_training( 
]
block_7_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_7_add/addAdd!block_6_project_BN/FusedBatchNorm!block_7_project_BN/FusedBatchNorm*
T0*/
_output_shapes
:���������@
�
6block_8_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   �  *(
_class
loc:@block_8_expand/kernel*
dtype0*
_output_shapes
:
�
4block_8_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *���*(
_class
loc:@block_8_expand/kernel*
dtype0*
_output_shapes
: 
�
4block_8_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=*(
_class
loc:@block_8_expand/kernel*
dtype0*
_output_shapes
: 
�
>block_8_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform6block_8_expand/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@block_8_expand/kernel*
dtype0*'
_output_shapes
:@�
�
4block_8_expand/kernel/Initializer/random_uniform/subSub4block_8_expand/kernel/Initializer/random_uniform/max4block_8_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_8_expand/kernel*
_output_shapes
: 
�
4block_8_expand/kernel/Initializer/random_uniform/mulMul>block_8_expand/kernel/Initializer/random_uniform/RandomUniform4block_8_expand/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@block_8_expand/kernel*'
_output_shapes
:@�
�
0block_8_expand/kernel/Initializer/random_uniformAdd4block_8_expand/kernel/Initializer/random_uniform/mul4block_8_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_8_expand/kernel*'
_output_shapes
:@�
�
block_8_expand/kernelVarHandleOp*
shape:@�*&
shared_nameblock_8_expand/kernel*(
_class
loc:@block_8_expand/kernel*
dtype0*
_output_shapes
: 
{
6block_8_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_8_expand/kernel*
_output_shapes
: 
�
block_8_expand/kernel/AssignAssignVariableOpblock_8_expand/kernel0block_8_expand/kernel/Initializer/random_uniform*(
_class
loc:@block_8_expand/kernel*
dtype0
�
)block_8_expand/kernel/Read/ReadVariableOpReadVariableOpblock_8_expand/kernel*(
_class
loc:@block_8_expand/kernel*
dtype0*'
_output_shapes
:@�
m
block_8_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
$block_8_expand/Conv2D/ReadVariableOpReadVariableOpblock_8_expand/kernel*
dtype0*'
_output_shapes
:@�
�
block_8_expand/Conv2DConv2Dblock_7_add/add$block_8_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
(block_8_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?**
_class 
loc:@block_8_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_8_expand_BN/gammaVarHandleOp*
shape:�*(
shared_nameblock_8_expand_BN/gamma**
_class 
loc:@block_8_expand_BN/gamma*
dtype0*
_output_shapes
: 

8block_8_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_8_expand_BN/gamma*
_output_shapes
: 
�
block_8_expand_BN/gamma/AssignAssignVariableOpblock_8_expand_BN/gamma(block_8_expand_BN/gamma/Initializer/ones**
_class 
loc:@block_8_expand_BN/gamma*
dtype0
�
+block_8_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_8_expand_BN/gamma**
_class 
loc:@block_8_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
(block_8_expand_BN/beta/Initializer/zerosConst*
valueB�*    *)
_class
loc:@block_8_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_8_expand_BN/betaVarHandleOp*
shape:�*'
shared_nameblock_8_expand_BN/beta*)
_class
loc:@block_8_expand_BN/beta*
dtype0*
_output_shapes
: 
}
7block_8_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_8_expand_BN/beta*
_output_shapes
: 
�
block_8_expand_BN/beta/AssignAssignVariableOpblock_8_expand_BN/beta(block_8_expand_BN/beta/Initializer/zeros*)
_class
loc:@block_8_expand_BN/beta*
dtype0
�
*block_8_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_8_expand_BN/beta*)
_class
loc:@block_8_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_8_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *0
_class&
$"loc:@block_8_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_8_expand_BN/moving_meanVarHandleOp*
shape:�*.
shared_nameblock_8_expand_BN/moving_mean*0
_class&
$"loc:@block_8_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
>block_8_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_8_expand_BN/moving_mean*
_output_shapes
: 
�
$block_8_expand_BN/moving_mean/AssignAssignVariableOpblock_8_expand_BN/moving_mean/block_8_expand_BN/moving_mean/Initializer/zeros*0
_class&
$"loc:@block_8_expand_BN/moving_mean*
dtype0
�
1block_8_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_8_expand_BN/moving_mean*0
_class&
$"loc:@block_8_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_8_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*4
_class*
(&loc:@block_8_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_8_expand_BN/moving_varianceVarHandleOp*
shape:�*2
shared_name#!block_8_expand_BN/moving_variance*4
_class*
(&loc:@block_8_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Bblock_8_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_8_expand_BN/moving_variance*
_output_shapes
: 
�
(block_8_expand_BN/moving_variance/AssignAssignVariableOp!block_8_expand_BN/moving_variance2block_8_expand_BN/moving_variance/Initializer/ones*4
_class*
(&loc:@block_8_expand_BN/moving_variance*
dtype0
�
5block_8_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp!block_8_expand_BN/moving_variance*4
_class*
(&loc:@block_8_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
u
 block_8_expand_BN/ReadVariableOpReadVariableOpblock_8_expand_BN/gamma*
dtype0*
_output_shapes	
:�
v
"block_8_expand_BN/ReadVariableOp_1ReadVariableOpblock_8_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_8_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_8_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
1block_8_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp!block_8_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
 block_8_expand_BN/FusedBatchNormFusedBatchNormblock_8_expand/Conv2D block_8_expand_BN/ReadVariableOp"block_8_expand_BN/ReadVariableOp_1/block_8_expand_BN/FusedBatchNorm/ReadVariableOp1block_8_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
\
block_8_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 

block_8_expand_relu/Relu6Relu6 block_8_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
Cblock_8_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      �     *5
_class+
)'loc:@block_8_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Ablock_8_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *�q*�*5
_class+
)'loc:@block_8_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Ablock_8_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *�q*=*5
_class+
)'loc:@block_8_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Kblock_8_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformCblock_8_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*5
_class+
)'loc:@block_8_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Ablock_8_depthwise/depthwise_kernel/Initializer/random_uniform/subSubAblock_8_depthwise/depthwise_kernel/Initializer/random_uniform/maxAblock_8_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_8_depthwise/depthwise_kernel*
_output_shapes
: 
�
Ablock_8_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulKblock_8_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformAblock_8_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@block_8_depthwise/depthwise_kernel*'
_output_shapes
:�
�
=block_8_depthwise/depthwise_kernel/Initializer/random_uniformAddAblock_8_depthwise/depthwise_kernel/Initializer/random_uniform/mulAblock_8_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_8_depthwise/depthwise_kernel*'
_output_shapes
:�
�
"block_8_depthwise/depthwise_kernelVarHandleOp*
shape:�*3
shared_name$"block_8_depthwise/depthwise_kernel*5
_class+
)'loc:@block_8_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Cblock_8_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_8_depthwise/depthwise_kernel*
_output_shapes
: 
�
)block_8_depthwise/depthwise_kernel/AssignAssignVariableOp"block_8_depthwise/depthwise_kernel=block_8_depthwise/depthwise_kernel/Initializer/random_uniform*5
_class+
)'loc:@block_8_depthwise/depthwise_kernel*
dtype0
�
6block_8_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp"block_8_depthwise/depthwise_kernel*5
_class+
)'loc:@block_8_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
*block_8_depthwise/depthwise/ReadVariableOpReadVariableOp"block_8_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
z
!block_8_depthwise/depthwise/ShapeConst*%
valueB"      �     *
dtype0*
_output_shapes
:
z
)block_8_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_8_depthwise/depthwiseDepthwiseConv2dNativeblock_8_expand_relu/Relu6*block_8_depthwise/depthwise/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
+block_8_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*-
_class#
!loc:@block_8_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_8_depthwise_BN/gammaVarHandleOp*
shape:�*+
shared_nameblock_8_depthwise_BN/gamma*-
_class#
!loc:@block_8_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
;block_8_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_8_depthwise_BN/gamma*
_output_shapes
: 
�
!block_8_depthwise_BN/gamma/AssignAssignVariableOpblock_8_depthwise_BN/gamma+block_8_depthwise_BN/gamma/Initializer/ones*-
_class#
!loc:@block_8_depthwise_BN/gamma*
dtype0
�
.block_8_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_8_depthwise_BN/gamma*-
_class#
!loc:@block_8_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
+block_8_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *,
_class"
 loc:@block_8_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_8_depthwise_BN/betaVarHandleOp*
shape:�**
shared_nameblock_8_depthwise_BN/beta*,
_class"
 loc:@block_8_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
:block_8_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_8_depthwise_BN/beta*
_output_shapes
: 
�
 block_8_depthwise_BN/beta/AssignAssignVariableOpblock_8_depthwise_BN/beta+block_8_depthwise_BN/beta/Initializer/zeros*,
_class"
 loc:@block_8_depthwise_BN/beta*
dtype0
�
-block_8_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_8_depthwise_BN/beta*,
_class"
 loc:@block_8_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_8_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *3
_class)
'%loc:@block_8_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
 block_8_depthwise_BN/moving_meanVarHandleOp*
shape:�*1
shared_name" block_8_depthwise_BN/moving_mean*3
_class)
'%loc:@block_8_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Ablock_8_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp block_8_depthwise_BN/moving_mean*
_output_shapes
: 
�
'block_8_depthwise_BN/moving_mean/AssignAssignVariableOp block_8_depthwise_BN/moving_mean2block_8_depthwise_BN/moving_mean/Initializer/zeros*3
_class)
'%loc:@block_8_depthwise_BN/moving_mean*
dtype0
�
4block_8_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp block_8_depthwise_BN/moving_mean*3
_class)
'%loc:@block_8_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_8_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*7
_class-
+)loc:@block_8_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_8_depthwise_BN/moving_varianceVarHandleOp*
shape:�*5
shared_name&$block_8_depthwise_BN/moving_variance*7
_class-
+)loc:@block_8_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Eblock_8_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp$block_8_depthwise_BN/moving_variance*
_output_shapes
: 
�
+block_8_depthwise_BN/moving_variance/AssignAssignVariableOp$block_8_depthwise_BN/moving_variance5block_8_depthwise_BN/moving_variance/Initializer/ones*7
_class-
+)loc:@block_8_depthwise_BN/moving_variance*
dtype0
�
8block_8_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp$block_8_depthwise_BN/moving_variance*7
_class-
+)loc:@block_8_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
{
#block_8_depthwise_BN/ReadVariableOpReadVariableOpblock_8_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
|
%block_8_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_8_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_8_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp block_8_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
4block_8_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp$block_8_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
#block_8_depthwise_BN/FusedBatchNormFusedBatchNormblock_8_depthwise/depthwise#block_8_depthwise_BN/ReadVariableOp%block_8_depthwise_BN/ReadVariableOp_12block_8_depthwise_BN/FusedBatchNorm/ReadVariableOp4block_8_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
_
block_8_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_8_depthwise_relu/Relu6Relu6#block_8_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
7block_8_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �  @   *)
_class
loc:@block_8_project/kernel*
dtype0*
_output_shapes
:
�
5block_8_project/kernel/Initializer/random_uniform/minConst*
valueB
 *���*)
_class
loc:@block_8_project/kernel*
dtype0*
_output_shapes
: 
�
5block_8_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=*)
_class
loc:@block_8_project/kernel*
dtype0*
_output_shapes
: 
�
?block_8_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_8_project/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_8_project/kernel*
dtype0*'
_output_shapes
:�@
�
5block_8_project/kernel/Initializer/random_uniform/subSub5block_8_project/kernel/Initializer/random_uniform/max5block_8_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_8_project/kernel*
_output_shapes
: 
�
5block_8_project/kernel/Initializer/random_uniform/mulMul?block_8_project/kernel/Initializer/random_uniform/RandomUniform5block_8_project/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_8_project/kernel*'
_output_shapes
:�@
�
1block_8_project/kernel/Initializer/random_uniformAdd5block_8_project/kernel/Initializer/random_uniform/mul5block_8_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_8_project/kernel*'
_output_shapes
:�@
�
block_8_project/kernelVarHandleOp*
shape:�@*'
shared_nameblock_8_project/kernel*)
_class
loc:@block_8_project/kernel*
dtype0*
_output_shapes
: 
}
7block_8_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_8_project/kernel*
_output_shapes
: 
�
block_8_project/kernel/AssignAssignVariableOpblock_8_project/kernel1block_8_project/kernel/Initializer/random_uniform*)
_class
loc:@block_8_project/kernel*
dtype0
�
*block_8_project/kernel/Read/ReadVariableOpReadVariableOpblock_8_project/kernel*)
_class
loc:@block_8_project/kernel*
dtype0*'
_output_shapes
:�@
n
block_8_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_8_project/Conv2D/ReadVariableOpReadVariableOpblock_8_project/kernel*
dtype0*'
_output_shapes
:�@
�
block_8_project/Conv2DConv2Dblock_8_depthwise_relu/Relu6%block_8_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:���������@
�
)block_8_project_BN/gamma/Initializer/onesConst*
valueB@*  �?*+
_class!
loc:@block_8_project_BN/gamma*
dtype0*
_output_shapes
:@
�
block_8_project_BN/gammaVarHandleOp*
shape:@*)
shared_nameblock_8_project_BN/gamma*+
_class!
loc:@block_8_project_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_8_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_8_project_BN/gamma*
_output_shapes
: 
�
block_8_project_BN/gamma/AssignAssignVariableOpblock_8_project_BN/gamma)block_8_project_BN/gamma/Initializer/ones*+
_class!
loc:@block_8_project_BN/gamma*
dtype0
�
,block_8_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_8_project_BN/gamma*+
_class!
loc:@block_8_project_BN/gamma*
dtype0*
_output_shapes
:@
�
)block_8_project_BN/beta/Initializer/zerosConst*
valueB@*    **
_class 
loc:@block_8_project_BN/beta*
dtype0*
_output_shapes
:@
�
block_8_project_BN/betaVarHandleOp*
shape:@*(
shared_nameblock_8_project_BN/beta**
_class 
loc:@block_8_project_BN/beta*
dtype0*
_output_shapes
: 

8block_8_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_8_project_BN/beta*
_output_shapes
: 
�
block_8_project_BN/beta/AssignAssignVariableOpblock_8_project_BN/beta)block_8_project_BN/beta/Initializer/zeros**
_class 
loc:@block_8_project_BN/beta*
dtype0
�
+block_8_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_8_project_BN/beta**
_class 
loc:@block_8_project_BN/beta*
dtype0*
_output_shapes
:@
�
0block_8_project_BN/moving_mean/Initializer/zerosConst*
valueB@*    *1
_class'
%#loc:@block_8_project_BN/moving_mean*
dtype0*
_output_shapes
:@
�
block_8_project_BN/moving_meanVarHandleOp*
shape:@*/
shared_name block_8_project_BN/moving_mean*1
_class'
%#loc:@block_8_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_8_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_8_project_BN/moving_mean*
_output_shapes
: 
�
%block_8_project_BN/moving_mean/AssignAssignVariableOpblock_8_project_BN/moving_mean0block_8_project_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_8_project_BN/moving_mean*
dtype0
�
2block_8_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_8_project_BN/moving_mean*1
_class'
%#loc:@block_8_project_BN/moving_mean*
dtype0*
_output_shapes
:@
�
3block_8_project_BN/moving_variance/Initializer/onesConst*
valueB@*  �?*5
_class+
)'loc:@block_8_project_BN/moving_variance*
dtype0*
_output_shapes
:@
�
"block_8_project_BN/moving_varianceVarHandleOp*
shape:@*3
shared_name$"block_8_project_BN/moving_variance*5
_class+
)'loc:@block_8_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_8_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_8_project_BN/moving_variance*
_output_shapes
: 
�
)block_8_project_BN/moving_variance/AssignAssignVariableOp"block_8_project_BN/moving_variance3block_8_project_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_8_project_BN/moving_variance*
dtype0
�
6block_8_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_8_project_BN/moving_variance*5
_class+
)'loc:@block_8_project_BN/moving_variance*
dtype0*
_output_shapes
:@
v
!block_8_project_BN/ReadVariableOpReadVariableOpblock_8_project_BN/gamma*
dtype0*
_output_shapes
:@
w
#block_8_project_BN/ReadVariableOp_1ReadVariableOpblock_8_project_BN/beta*
dtype0*
_output_shapes
:@
�
0block_8_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_8_project_BN/moving_mean*
dtype0*
_output_shapes
:@
�
2block_8_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_8_project_BN/moving_variance*
dtype0*
_output_shapes
:@
�
!block_8_project_BN/FusedBatchNormFusedBatchNormblock_8_project/Conv2D!block_8_project_BN/ReadVariableOp#block_8_project_BN/ReadVariableOp_10block_8_project_BN/FusedBatchNorm/ReadVariableOp2block_8_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:���������@:@:@:@:@*
is_training( 
]
block_8_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_8_add/addAddblock_7_add/add!block_8_project_BN/FusedBatchNorm*
T0*/
_output_shapes
:���������@
�
6block_9_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   �  *(
_class
loc:@block_9_expand/kernel*
dtype0*
_output_shapes
:
�
4block_9_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *���*(
_class
loc:@block_9_expand/kernel*
dtype0*
_output_shapes
: 
�
4block_9_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=*(
_class
loc:@block_9_expand/kernel*
dtype0*
_output_shapes
: 
�
>block_9_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform6block_9_expand/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@block_9_expand/kernel*
dtype0*'
_output_shapes
:@�
�
4block_9_expand/kernel/Initializer/random_uniform/subSub4block_9_expand/kernel/Initializer/random_uniform/max4block_9_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_9_expand/kernel*
_output_shapes
: 
�
4block_9_expand/kernel/Initializer/random_uniform/mulMul>block_9_expand/kernel/Initializer/random_uniform/RandomUniform4block_9_expand/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@block_9_expand/kernel*'
_output_shapes
:@�
�
0block_9_expand/kernel/Initializer/random_uniformAdd4block_9_expand/kernel/Initializer/random_uniform/mul4block_9_expand/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@block_9_expand/kernel*'
_output_shapes
:@�
�
block_9_expand/kernelVarHandleOp*
shape:@�*&
shared_nameblock_9_expand/kernel*(
_class
loc:@block_9_expand/kernel*
dtype0*
_output_shapes
: 
{
6block_9_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_9_expand/kernel*
_output_shapes
: 
�
block_9_expand/kernel/AssignAssignVariableOpblock_9_expand/kernel0block_9_expand/kernel/Initializer/random_uniform*(
_class
loc:@block_9_expand/kernel*
dtype0
�
)block_9_expand/kernel/Read/ReadVariableOpReadVariableOpblock_9_expand/kernel*(
_class
loc:@block_9_expand/kernel*
dtype0*'
_output_shapes
:@�
m
block_9_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
$block_9_expand/Conv2D/ReadVariableOpReadVariableOpblock_9_expand/kernel*
dtype0*'
_output_shapes
:@�
�
block_9_expand/Conv2DConv2Dblock_8_add/add$block_9_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
(block_9_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?**
_class 
loc:@block_9_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_9_expand_BN/gammaVarHandleOp*
shape:�*(
shared_nameblock_9_expand_BN/gamma**
_class 
loc:@block_9_expand_BN/gamma*
dtype0*
_output_shapes
: 

8block_9_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_9_expand_BN/gamma*
_output_shapes
: 
�
block_9_expand_BN/gamma/AssignAssignVariableOpblock_9_expand_BN/gamma(block_9_expand_BN/gamma/Initializer/ones**
_class 
loc:@block_9_expand_BN/gamma*
dtype0
�
+block_9_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_9_expand_BN/gamma**
_class 
loc:@block_9_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
(block_9_expand_BN/beta/Initializer/zerosConst*
valueB�*    *)
_class
loc:@block_9_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_9_expand_BN/betaVarHandleOp*
shape:�*'
shared_nameblock_9_expand_BN/beta*)
_class
loc:@block_9_expand_BN/beta*
dtype0*
_output_shapes
: 
}
7block_9_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_9_expand_BN/beta*
_output_shapes
: 
�
block_9_expand_BN/beta/AssignAssignVariableOpblock_9_expand_BN/beta(block_9_expand_BN/beta/Initializer/zeros*)
_class
loc:@block_9_expand_BN/beta*
dtype0
�
*block_9_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_9_expand_BN/beta*)
_class
loc:@block_9_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_9_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *0
_class&
$"loc:@block_9_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_9_expand_BN/moving_meanVarHandleOp*
shape:�*.
shared_nameblock_9_expand_BN/moving_mean*0
_class&
$"loc:@block_9_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
>block_9_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_9_expand_BN/moving_mean*
_output_shapes
: 
�
$block_9_expand_BN/moving_mean/AssignAssignVariableOpblock_9_expand_BN/moving_mean/block_9_expand_BN/moving_mean/Initializer/zeros*0
_class&
$"loc:@block_9_expand_BN/moving_mean*
dtype0
�
1block_9_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_9_expand_BN/moving_mean*0
_class&
$"loc:@block_9_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_9_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*4
_class*
(&loc:@block_9_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_9_expand_BN/moving_varianceVarHandleOp*
shape:�*2
shared_name#!block_9_expand_BN/moving_variance*4
_class*
(&loc:@block_9_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Bblock_9_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_9_expand_BN/moving_variance*
_output_shapes
: 
�
(block_9_expand_BN/moving_variance/AssignAssignVariableOp!block_9_expand_BN/moving_variance2block_9_expand_BN/moving_variance/Initializer/ones*4
_class*
(&loc:@block_9_expand_BN/moving_variance*
dtype0
�
5block_9_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp!block_9_expand_BN/moving_variance*4
_class*
(&loc:@block_9_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
u
 block_9_expand_BN/ReadVariableOpReadVariableOpblock_9_expand_BN/gamma*
dtype0*
_output_shapes	
:�
v
"block_9_expand_BN/ReadVariableOp_1ReadVariableOpblock_9_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
/block_9_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_9_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
1block_9_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp!block_9_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
 block_9_expand_BN/FusedBatchNormFusedBatchNormblock_9_expand/Conv2D block_9_expand_BN/ReadVariableOp"block_9_expand_BN/ReadVariableOp_1/block_9_expand_BN/FusedBatchNorm/ReadVariableOp1block_9_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
\
block_9_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 

block_9_expand_relu/Relu6Relu6 block_9_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
Cblock_9_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      �     *5
_class+
)'loc:@block_9_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Ablock_9_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *�q*�*5
_class+
)'loc:@block_9_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Ablock_9_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *�q*=*5
_class+
)'loc:@block_9_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Kblock_9_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformCblock_9_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*5
_class+
)'loc:@block_9_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Ablock_9_depthwise/depthwise_kernel/Initializer/random_uniform/subSubAblock_9_depthwise/depthwise_kernel/Initializer/random_uniform/maxAblock_9_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_9_depthwise/depthwise_kernel*
_output_shapes
: 
�
Ablock_9_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulKblock_9_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformAblock_9_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@block_9_depthwise/depthwise_kernel*'
_output_shapes
:�
�
=block_9_depthwise/depthwise_kernel/Initializer/random_uniformAddAblock_9_depthwise/depthwise_kernel/Initializer/random_uniform/mulAblock_9_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@block_9_depthwise/depthwise_kernel*'
_output_shapes
:�
�
"block_9_depthwise/depthwise_kernelVarHandleOp*
shape:�*3
shared_name$"block_9_depthwise/depthwise_kernel*5
_class+
)'loc:@block_9_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Cblock_9_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_9_depthwise/depthwise_kernel*
_output_shapes
: 
�
)block_9_depthwise/depthwise_kernel/AssignAssignVariableOp"block_9_depthwise/depthwise_kernel=block_9_depthwise/depthwise_kernel/Initializer/random_uniform*5
_class+
)'loc:@block_9_depthwise/depthwise_kernel*
dtype0
�
6block_9_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp"block_9_depthwise/depthwise_kernel*5
_class+
)'loc:@block_9_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
*block_9_depthwise/depthwise/ReadVariableOpReadVariableOp"block_9_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
z
!block_9_depthwise/depthwise/ShapeConst*%
valueB"      �     *
dtype0*
_output_shapes
:
z
)block_9_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_9_depthwise/depthwiseDepthwiseConv2dNativeblock_9_expand_relu/Relu6*block_9_depthwise/depthwise/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
+block_9_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*-
_class#
!loc:@block_9_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_9_depthwise_BN/gammaVarHandleOp*
shape:�*+
shared_nameblock_9_depthwise_BN/gamma*-
_class#
!loc:@block_9_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
;block_9_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_9_depthwise_BN/gamma*
_output_shapes
: 
�
!block_9_depthwise_BN/gamma/AssignAssignVariableOpblock_9_depthwise_BN/gamma+block_9_depthwise_BN/gamma/Initializer/ones*-
_class#
!loc:@block_9_depthwise_BN/gamma*
dtype0
�
.block_9_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_9_depthwise_BN/gamma*-
_class#
!loc:@block_9_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
+block_9_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *,
_class"
 loc:@block_9_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_9_depthwise_BN/betaVarHandleOp*
shape:�**
shared_nameblock_9_depthwise_BN/beta*,
_class"
 loc:@block_9_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
:block_9_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_9_depthwise_BN/beta*
_output_shapes
: 
�
 block_9_depthwise_BN/beta/AssignAssignVariableOpblock_9_depthwise_BN/beta+block_9_depthwise_BN/beta/Initializer/zeros*,
_class"
 loc:@block_9_depthwise_BN/beta*
dtype0
�
-block_9_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_9_depthwise_BN/beta*,
_class"
 loc:@block_9_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_9_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *3
_class)
'%loc:@block_9_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
 block_9_depthwise_BN/moving_meanVarHandleOp*
shape:�*1
shared_name" block_9_depthwise_BN/moving_mean*3
_class)
'%loc:@block_9_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Ablock_9_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp block_9_depthwise_BN/moving_mean*
_output_shapes
: 
�
'block_9_depthwise_BN/moving_mean/AssignAssignVariableOp block_9_depthwise_BN/moving_mean2block_9_depthwise_BN/moving_mean/Initializer/zeros*3
_class)
'%loc:@block_9_depthwise_BN/moving_mean*
dtype0
�
4block_9_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp block_9_depthwise_BN/moving_mean*3
_class)
'%loc:@block_9_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_9_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*7
_class-
+)loc:@block_9_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_9_depthwise_BN/moving_varianceVarHandleOp*
shape:�*5
shared_name&$block_9_depthwise_BN/moving_variance*7
_class-
+)loc:@block_9_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Eblock_9_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp$block_9_depthwise_BN/moving_variance*
_output_shapes
: 
�
+block_9_depthwise_BN/moving_variance/AssignAssignVariableOp$block_9_depthwise_BN/moving_variance5block_9_depthwise_BN/moving_variance/Initializer/ones*7
_class-
+)loc:@block_9_depthwise_BN/moving_variance*
dtype0
�
8block_9_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp$block_9_depthwise_BN/moving_variance*7
_class-
+)loc:@block_9_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
{
#block_9_depthwise_BN/ReadVariableOpReadVariableOpblock_9_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
|
%block_9_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_9_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
2block_9_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp block_9_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
4block_9_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp$block_9_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
#block_9_depthwise_BN/FusedBatchNormFusedBatchNormblock_9_depthwise/depthwise#block_9_depthwise_BN/ReadVariableOp%block_9_depthwise_BN/ReadVariableOp_12block_9_depthwise_BN/FusedBatchNorm/ReadVariableOp4block_9_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
_
block_9_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_9_depthwise_relu/Relu6Relu6#block_9_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
7block_9_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �  @   *)
_class
loc:@block_9_project/kernel*
dtype0*
_output_shapes
:
�
5block_9_project/kernel/Initializer/random_uniform/minConst*
valueB
 *���*)
_class
loc:@block_9_project/kernel*
dtype0*
_output_shapes
: 
�
5block_9_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=*)
_class
loc:@block_9_project/kernel*
dtype0*
_output_shapes
: 
�
?block_9_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_9_project/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_9_project/kernel*
dtype0*'
_output_shapes
:�@
�
5block_9_project/kernel/Initializer/random_uniform/subSub5block_9_project/kernel/Initializer/random_uniform/max5block_9_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_9_project/kernel*
_output_shapes
: 
�
5block_9_project/kernel/Initializer/random_uniform/mulMul?block_9_project/kernel/Initializer/random_uniform/RandomUniform5block_9_project/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_9_project/kernel*'
_output_shapes
:�@
�
1block_9_project/kernel/Initializer/random_uniformAdd5block_9_project/kernel/Initializer/random_uniform/mul5block_9_project/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_9_project/kernel*'
_output_shapes
:�@
�
block_9_project/kernelVarHandleOp*
shape:�@*'
shared_nameblock_9_project/kernel*)
_class
loc:@block_9_project/kernel*
dtype0*
_output_shapes
: 
}
7block_9_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_9_project/kernel*
_output_shapes
: 
�
block_9_project/kernel/AssignAssignVariableOpblock_9_project/kernel1block_9_project/kernel/Initializer/random_uniform*)
_class
loc:@block_9_project/kernel*
dtype0
�
*block_9_project/kernel/Read/ReadVariableOpReadVariableOpblock_9_project/kernel*)
_class
loc:@block_9_project/kernel*
dtype0*'
_output_shapes
:�@
n
block_9_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_9_project/Conv2D/ReadVariableOpReadVariableOpblock_9_project/kernel*
dtype0*'
_output_shapes
:�@
�
block_9_project/Conv2DConv2Dblock_9_depthwise_relu/Relu6%block_9_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:���������@
�
)block_9_project_BN/gamma/Initializer/onesConst*
valueB@*  �?*+
_class!
loc:@block_9_project_BN/gamma*
dtype0*
_output_shapes
:@
�
block_9_project_BN/gammaVarHandleOp*
shape:@*)
shared_nameblock_9_project_BN/gamma*+
_class!
loc:@block_9_project_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_9_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_9_project_BN/gamma*
_output_shapes
: 
�
block_9_project_BN/gamma/AssignAssignVariableOpblock_9_project_BN/gamma)block_9_project_BN/gamma/Initializer/ones*+
_class!
loc:@block_9_project_BN/gamma*
dtype0
�
,block_9_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_9_project_BN/gamma*+
_class!
loc:@block_9_project_BN/gamma*
dtype0*
_output_shapes
:@
�
)block_9_project_BN/beta/Initializer/zerosConst*
valueB@*    **
_class 
loc:@block_9_project_BN/beta*
dtype0*
_output_shapes
:@
�
block_9_project_BN/betaVarHandleOp*
shape:@*(
shared_nameblock_9_project_BN/beta**
_class 
loc:@block_9_project_BN/beta*
dtype0*
_output_shapes
: 

8block_9_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_9_project_BN/beta*
_output_shapes
: 
�
block_9_project_BN/beta/AssignAssignVariableOpblock_9_project_BN/beta)block_9_project_BN/beta/Initializer/zeros**
_class 
loc:@block_9_project_BN/beta*
dtype0
�
+block_9_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_9_project_BN/beta**
_class 
loc:@block_9_project_BN/beta*
dtype0*
_output_shapes
:@
�
0block_9_project_BN/moving_mean/Initializer/zerosConst*
valueB@*    *1
_class'
%#loc:@block_9_project_BN/moving_mean*
dtype0*
_output_shapes
:@
�
block_9_project_BN/moving_meanVarHandleOp*
shape:@*/
shared_name block_9_project_BN/moving_mean*1
_class'
%#loc:@block_9_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_9_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_9_project_BN/moving_mean*
_output_shapes
: 
�
%block_9_project_BN/moving_mean/AssignAssignVariableOpblock_9_project_BN/moving_mean0block_9_project_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_9_project_BN/moving_mean*
dtype0
�
2block_9_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_9_project_BN/moving_mean*1
_class'
%#loc:@block_9_project_BN/moving_mean*
dtype0*
_output_shapes
:@
�
3block_9_project_BN/moving_variance/Initializer/onesConst*
valueB@*  �?*5
_class+
)'loc:@block_9_project_BN/moving_variance*
dtype0*
_output_shapes
:@
�
"block_9_project_BN/moving_varianceVarHandleOp*
shape:@*3
shared_name$"block_9_project_BN/moving_variance*5
_class+
)'loc:@block_9_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_9_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_9_project_BN/moving_variance*
_output_shapes
: 
�
)block_9_project_BN/moving_variance/AssignAssignVariableOp"block_9_project_BN/moving_variance3block_9_project_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_9_project_BN/moving_variance*
dtype0
�
6block_9_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_9_project_BN/moving_variance*5
_class+
)'loc:@block_9_project_BN/moving_variance*
dtype0*
_output_shapes
:@
v
!block_9_project_BN/ReadVariableOpReadVariableOpblock_9_project_BN/gamma*
dtype0*
_output_shapes
:@
w
#block_9_project_BN/ReadVariableOp_1ReadVariableOpblock_9_project_BN/beta*
dtype0*
_output_shapes
:@
�
0block_9_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_9_project_BN/moving_mean*
dtype0*
_output_shapes
:@
�
2block_9_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_9_project_BN/moving_variance*
dtype0*
_output_shapes
:@
�
!block_9_project_BN/FusedBatchNormFusedBatchNormblock_9_project/Conv2D!block_9_project_BN/ReadVariableOp#block_9_project_BN/ReadVariableOp_10block_9_project_BN/FusedBatchNorm/ReadVariableOp2block_9_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:���������@:@:@:@:@*
is_training( 
]
block_9_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_9_add/addAddblock_8_add/add!block_9_project_BN/FusedBatchNorm*
T0*/
_output_shapes
:���������@
�
7block_10_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   �  *)
_class
loc:@block_10_expand/kernel*
dtype0*
_output_shapes
:
�
5block_10_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *���*)
_class
loc:@block_10_expand/kernel*
dtype0*
_output_shapes
: 
�
5block_10_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=*)
_class
loc:@block_10_expand/kernel*
dtype0*
_output_shapes
: 
�
?block_10_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_10_expand/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_10_expand/kernel*
dtype0*'
_output_shapes
:@�
�
5block_10_expand/kernel/Initializer/random_uniform/subSub5block_10_expand/kernel/Initializer/random_uniform/max5block_10_expand/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_10_expand/kernel*
_output_shapes
: 
�
5block_10_expand/kernel/Initializer/random_uniform/mulMul?block_10_expand/kernel/Initializer/random_uniform/RandomUniform5block_10_expand/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_10_expand/kernel*'
_output_shapes
:@�
�
1block_10_expand/kernel/Initializer/random_uniformAdd5block_10_expand/kernel/Initializer/random_uniform/mul5block_10_expand/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_10_expand/kernel*'
_output_shapes
:@�
�
block_10_expand/kernelVarHandleOp*
shape:@�*'
shared_nameblock_10_expand/kernel*)
_class
loc:@block_10_expand/kernel*
dtype0*
_output_shapes
: 
}
7block_10_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_10_expand/kernel*
_output_shapes
: 
�
block_10_expand/kernel/AssignAssignVariableOpblock_10_expand/kernel1block_10_expand/kernel/Initializer/random_uniform*)
_class
loc:@block_10_expand/kernel*
dtype0
�
*block_10_expand/kernel/Read/ReadVariableOpReadVariableOpblock_10_expand/kernel*)
_class
loc:@block_10_expand/kernel*
dtype0*'
_output_shapes
:@�
n
block_10_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_10_expand/Conv2D/ReadVariableOpReadVariableOpblock_10_expand/kernel*
dtype0*'
_output_shapes
:@�
�
block_10_expand/Conv2DConv2Dblock_9_add/add%block_10_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
)block_10_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?*+
_class!
loc:@block_10_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_10_expand_BN/gammaVarHandleOp*
shape:�*)
shared_nameblock_10_expand_BN/gamma*+
_class!
loc:@block_10_expand_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_10_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_10_expand_BN/gamma*
_output_shapes
: 
�
block_10_expand_BN/gamma/AssignAssignVariableOpblock_10_expand_BN/gamma)block_10_expand_BN/gamma/Initializer/ones*+
_class!
loc:@block_10_expand_BN/gamma*
dtype0
�
,block_10_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_10_expand_BN/gamma*+
_class!
loc:@block_10_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
)block_10_expand_BN/beta/Initializer/zerosConst*
valueB�*    **
_class 
loc:@block_10_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_10_expand_BN/betaVarHandleOp*
shape:�*(
shared_nameblock_10_expand_BN/beta**
_class 
loc:@block_10_expand_BN/beta*
dtype0*
_output_shapes
: 

8block_10_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_10_expand_BN/beta*
_output_shapes
: 
�
block_10_expand_BN/beta/AssignAssignVariableOpblock_10_expand_BN/beta)block_10_expand_BN/beta/Initializer/zeros**
_class 
loc:@block_10_expand_BN/beta*
dtype0
�
+block_10_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_10_expand_BN/beta**
_class 
loc:@block_10_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
0block_10_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *1
_class'
%#loc:@block_10_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_10_expand_BN/moving_meanVarHandleOp*
shape:�*/
shared_name block_10_expand_BN/moving_mean*1
_class'
%#loc:@block_10_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_10_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_10_expand_BN/moving_mean*
_output_shapes
: 
�
%block_10_expand_BN/moving_mean/AssignAssignVariableOpblock_10_expand_BN/moving_mean0block_10_expand_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_10_expand_BN/moving_mean*
dtype0
�
2block_10_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_10_expand_BN/moving_mean*1
_class'
%#loc:@block_10_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
3block_10_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*5
_class+
)'loc:@block_10_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
"block_10_expand_BN/moving_varianceVarHandleOp*
shape:�*3
shared_name$"block_10_expand_BN/moving_variance*5
_class+
)'loc:@block_10_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_10_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_10_expand_BN/moving_variance*
_output_shapes
: 
�
)block_10_expand_BN/moving_variance/AssignAssignVariableOp"block_10_expand_BN/moving_variance3block_10_expand_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_10_expand_BN/moving_variance*
dtype0
�
6block_10_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_10_expand_BN/moving_variance*5
_class+
)'loc:@block_10_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
w
!block_10_expand_BN/ReadVariableOpReadVariableOpblock_10_expand_BN/gamma*
dtype0*
_output_shapes	
:�
x
#block_10_expand_BN/ReadVariableOp_1ReadVariableOpblock_10_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
0block_10_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_10_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_10_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_10_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_10_expand_BN/FusedBatchNormFusedBatchNormblock_10_expand/Conv2D!block_10_expand_BN/ReadVariableOp#block_10_expand_BN/ReadVariableOp_10block_10_expand_BN/FusedBatchNorm/ReadVariableOp2block_10_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
]
block_10_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_10_expand_relu/Relu6Relu6!block_10_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
Dblock_10_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      �     *6
_class,
*(loc:@block_10_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Bblock_10_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *�q*�*6
_class,
*(loc:@block_10_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Bblock_10_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *�q*=*6
_class,
*(loc:@block_10_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Lblock_10_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformDblock_10_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@block_10_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Bblock_10_depthwise/depthwise_kernel/Initializer/random_uniform/subSubBblock_10_depthwise/depthwise_kernel/Initializer/random_uniform/maxBblock_10_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@block_10_depthwise/depthwise_kernel*
_output_shapes
: 
�
Bblock_10_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulLblock_10_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformBblock_10_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@block_10_depthwise/depthwise_kernel*'
_output_shapes
:�
�
>block_10_depthwise/depthwise_kernel/Initializer/random_uniformAddBblock_10_depthwise/depthwise_kernel/Initializer/random_uniform/mulBblock_10_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@block_10_depthwise/depthwise_kernel*'
_output_shapes
:�
�
#block_10_depthwise/depthwise_kernelVarHandleOp*
shape:�*4
shared_name%#block_10_depthwise/depthwise_kernel*6
_class,
*(loc:@block_10_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Dblock_10_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp#block_10_depthwise/depthwise_kernel*
_output_shapes
: 
�
*block_10_depthwise/depthwise_kernel/AssignAssignVariableOp#block_10_depthwise/depthwise_kernel>block_10_depthwise/depthwise_kernel/Initializer/random_uniform*6
_class,
*(loc:@block_10_depthwise/depthwise_kernel*
dtype0
�
7block_10_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp#block_10_depthwise/depthwise_kernel*6
_class,
*(loc:@block_10_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
+block_10_depthwise/depthwise/ReadVariableOpReadVariableOp#block_10_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
{
"block_10_depthwise/depthwise/ShapeConst*%
valueB"      �     *
dtype0*
_output_shapes
:
{
*block_10_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_10_depthwise/depthwiseDepthwiseConv2dNativeblock_10_expand_relu/Relu6+block_10_depthwise/depthwise/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
,block_10_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*.
_class$
" loc:@block_10_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_10_depthwise_BN/gammaVarHandleOp*
shape:�*,
shared_nameblock_10_depthwise_BN/gamma*.
_class$
" loc:@block_10_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
<block_10_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_10_depthwise_BN/gamma*
_output_shapes
: 
�
"block_10_depthwise_BN/gamma/AssignAssignVariableOpblock_10_depthwise_BN/gamma,block_10_depthwise_BN/gamma/Initializer/ones*.
_class$
" loc:@block_10_depthwise_BN/gamma*
dtype0
�
/block_10_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_10_depthwise_BN/gamma*.
_class$
" loc:@block_10_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
,block_10_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *-
_class#
!loc:@block_10_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_10_depthwise_BN/betaVarHandleOp*
shape:�*+
shared_nameblock_10_depthwise_BN/beta*-
_class#
!loc:@block_10_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
;block_10_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_10_depthwise_BN/beta*
_output_shapes
: 
�
!block_10_depthwise_BN/beta/AssignAssignVariableOpblock_10_depthwise_BN/beta,block_10_depthwise_BN/beta/Initializer/zeros*-
_class#
!loc:@block_10_depthwise_BN/beta*
dtype0
�
.block_10_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_10_depthwise_BN/beta*-
_class#
!loc:@block_10_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
3block_10_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *4
_class*
(&loc:@block_10_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
!block_10_depthwise_BN/moving_meanVarHandleOp*
shape:�*2
shared_name#!block_10_depthwise_BN/moving_mean*4
_class*
(&loc:@block_10_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Bblock_10_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_10_depthwise_BN/moving_mean*
_output_shapes
: 
�
(block_10_depthwise_BN/moving_mean/AssignAssignVariableOp!block_10_depthwise_BN/moving_mean3block_10_depthwise_BN/moving_mean/Initializer/zeros*4
_class*
(&loc:@block_10_depthwise_BN/moving_mean*
dtype0
�
5block_10_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp!block_10_depthwise_BN/moving_mean*4
_class*
(&loc:@block_10_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
6block_10_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*8
_class.
,*loc:@block_10_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
%block_10_depthwise_BN/moving_varianceVarHandleOp*
shape:�*6
shared_name'%block_10_depthwise_BN/moving_variance*8
_class.
,*loc:@block_10_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Fblock_10_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%block_10_depthwise_BN/moving_variance*
_output_shapes
: 
�
,block_10_depthwise_BN/moving_variance/AssignAssignVariableOp%block_10_depthwise_BN/moving_variance6block_10_depthwise_BN/moving_variance/Initializer/ones*8
_class.
,*loc:@block_10_depthwise_BN/moving_variance*
dtype0
�
9block_10_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp%block_10_depthwise_BN/moving_variance*8
_class.
,*loc:@block_10_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
}
$block_10_depthwise_BN/ReadVariableOpReadVariableOpblock_10_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
~
&block_10_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_10_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
3block_10_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp!block_10_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_10_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp%block_10_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_10_depthwise_BN/FusedBatchNormFusedBatchNormblock_10_depthwise/depthwise$block_10_depthwise_BN/ReadVariableOp&block_10_depthwise_BN/ReadVariableOp_13block_10_depthwise_BN/FusedBatchNorm/ReadVariableOp5block_10_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
`
block_10_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_10_depthwise_relu/Relu6Relu6$block_10_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
8block_10_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �  `   **
_class 
loc:@block_10_project/kernel*
dtype0*
_output_shapes
:
�
6block_10_project/kernel/Initializer/random_uniform/minConst*
valueB
 *.��**
_class 
loc:@block_10_project/kernel*
dtype0*
_output_shapes
: 
�
6block_10_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *.��=**
_class 
loc:@block_10_project/kernel*
dtype0*
_output_shapes
: 
�
@block_10_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform8block_10_project/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@block_10_project/kernel*
dtype0*'
_output_shapes
:�`
�
6block_10_project/kernel/Initializer/random_uniform/subSub6block_10_project/kernel/Initializer/random_uniform/max6block_10_project/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@block_10_project/kernel*
_output_shapes
: 
�
6block_10_project/kernel/Initializer/random_uniform/mulMul@block_10_project/kernel/Initializer/random_uniform/RandomUniform6block_10_project/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@block_10_project/kernel*'
_output_shapes
:�`
�
2block_10_project/kernel/Initializer/random_uniformAdd6block_10_project/kernel/Initializer/random_uniform/mul6block_10_project/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@block_10_project/kernel*'
_output_shapes
:�`
�
block_10_project/kernelVarHandleOp*
shape:�`*(
shared_nameblock_10_project/kernel**
_class 
loc:@block_10_project/kernel*
dtype0*
_output_shapes
: 

8block_10_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_10_project/kernel*
_output_shapes
: 
�
block_10_project/kernel/AssignAssignVariableOpblock_10_project/kernel2block_10_project/kernel/Initializer/random_uniform**
_class 
loc:@block_10_project/kernel*
dtype0
�
+block_10_project/kernel/Read/ReadVariableOpReadVariableOpblock_10_project/kernel**
_class 
loc:@block_10_project/kernel*
dtype0*'
_output_shapes
:�`
o
block_10_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
&block_10_project/Conv2D/ReadVariableOpReadVariableOpblock_10_project/kernel*
dtype0*'
_output_shapes
:�`
�
block_10_project/Conv2DConv2Dblock_10_depthwise_relu/Relu6&block_10_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:���������`
�
*block_10_project_BN/gamma/Initializer/onesConst*
valueB`*  �?*,
_class"
 loc:@block_10_project_BN/gamma*
dtype0*
_output_shapes
:`
�
block_10_project_BN/gammaVarHandleOp*
shape:`**
shared_nameblock_10_project_BN/gamma*,
_class"
 loc:@block_10_project_BN/gamma*
dtype0*
_output_shapes
: 
�
:block_10_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_10_project_BN/gamma*
_output_shapes
: 
�
 block_10_project_BN/gamma/AssignAssignVariableOpblock_10_project_BN/gamma*block_10_project_BN/gamma/Initializer/ones*,
_class"
 loc:@block_10_project_BN/gamma*
dtype0
�
-block_10_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_10_project_BN/gamma*,
_class"
 loc:@block_10_project_BN/gamma*
dtype0*
_output_shapes
:`
�
*block_10_project_BN/beta/Initializer/zerosConst*
valueB`*    *+
_class!
loc:@block_10_project_BN/beta*
dtype0*
_output_shapes
:`
�
block_10_project_BN/betaVarHandleOp*
shape:`*)
shared_nameblock_10_project_BN/beta*+
_class!
loc:@block_10_project_BN/beta*
dtype0*
_output_shapes
: 
�
9block_10_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_10_project_BN/beta*
_output_shapes
: 
�
block_10_project_BN/beta/AssignAssignVariableOpblock_10_project_BN/beta*block_10_project_BN/beta/Initializer/zeros*+
_class!
loc:@block_10_project_BN/beta*
dtype0
�
,block_10_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_10_project_BN/beta*+
_class!
loc:@block_10_project_BN/beta*
dtype0*
_output_shapes
:`
�
1block_10_project_BN/moving_mean/Initializer/zerosConst*
valueB`*    *2
_class(
&$loc:@block_10_project_BN/moving_mean*
dtype0*
_output_shapes
:`
�
block_10_project_BN/moving_meanVarHandleOp*
shape:`*0
shared_name!block_10_project_BN/moving_mean*2
_class(
&$loc:@block_10_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
@block_10_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_10_project_BN/moving_mean*
_output_shapes
: 
�
&block_10_project_BN/moving_mean/AssignAssignVariableOpblock_10_project_BN/moving_mean1block_10_project_BN/moving_mean/Initializer/zeros*2
_class(
&$loc:@block_10_project_BN/moving_mean*
dtype0
�
3block_10_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_10_project_BN/moving_mean*2
_class(
&$loc:@block_10_project_BN/moving_mean*
dtype0*
_output_shapes
:`
�
4block_10_project_BN/moving_variance/Initializer/onesConst*
valueB`*  �?*6
_class,
*(loc:@block_10_project_BN/moving_variance*
dtype0*
_output_shapes
:`
�
#block_10_project_BN/moving_varianceVarHandleOp*
shape:`*4
shared_name%#block_10_project_BN/moving_variance*6
_class,
*(loc:@block_10_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Dblock_10_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#block_10_project_BN/moving_variance*
_output_shapes
: 
�
*block_10_project_BN/moving_variance/AssignAssignVariableOp#block_10_project_BN/moving_variance4block_10_project_BN/moving_variance/Initializer/ones*6
_class,
*(loc:@block_10_project_BN/moving_variance*
dtype0
�
7block_10_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp#block_10_project_BN/moving_variance*6
_class,
*(loc:@block_10_project_BN/moving_variance*
dtype0*
_output_shapes
:`
x
"block_10_project_BN/ReadVariableOpReadVariableOpblock_10_project_BN/gamma*
dtype0*
_output_shapes
:`
y
$block_10_project_BN/ReadVariableOp_1ReadVariableOpblock_10_project_BN/beta*
dtype0*
_output_shapes
:`
�
1block_10_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_10_project_BN/moving_mean*
dtype0*
_output_shapes
:`
�
3block_10_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp#block_10_project_BN/moving_variance*
dtype0*
_output_shapes
:`
�
"block_10_project_BN/FusedBatchNormFusedBatchNormblock_10_project/Conv2D"block_10_project_BN/ReadVariableOp$block_10_project_BN/ReadVariableOp_11block_10_project_BN/FusedBatchNorm/ReadVariableOp3block_10_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:���������`:`:`:`:`*
is_training( 
^
block_10_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
7block_11_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"      `   @  *)
_class
loc:@block_11_expand/kernel*
dtype0*
_output_shapes
:
�
5block_11_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *����*)
_class
loc:@block_11_expand/kernel*
dtype0*
_output_shapes
: 
�
5block_11_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *���=*)
_class
loc:@block_11_expand/kernel*
dtype0*
_output_shapes
: 
�
?block_11_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_11_expand/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_11_expand/kernel*
dtype0*'
_output_shapes
:`�
�
5block_11_expand/kernel/Initializer/random_uniform/subSub5block_11_expand/kernel/Initializer/random_uniform/max5block_11_expand/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_11_expand/kernel*
_output_shapes
: 
�
5block_11_expand/kernel/Initializer/random_uniform/mulMul?block_11_expand/kernel/Initializer/random_uniform/RandomUniform5block_11_expand/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_11_expand/kernel*'
_output_shapes
:`�
�
1block_11_expand/kernel/Initializer/random_uniformAdd5block_11_expand/kernel/Initializer/random_uniform/mul5block_11_expand/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_11_expand/kernel*'
_output_shapes
:`�
�
block_11_expand/kernelVarHandleOp*
shape:`�*'
shared_nameblock_11_expand/kernel*)
_class
loc:@block_11_expand/kernel*
dtype0*
_output_shapes
: 
}
7block_11_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_11_expand/kernel*
_output_shapes
: 
�
block_11_expand/kernel/AssignAssignVariableOpblock_11_expand/kernel1block_11_expand/kernel/Initializer/random_uniform*)
_class
loc:@block_11_expand/kernel*
dtype0
�
*block_11_expand/kernel/Read/ReadVariableOpReadVariableOpblock_11_expand/kernel*)
_class
loc:@block_11_expand/kernel*
dtype0*'
_output_shapes
:`�
n
block_11_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_11_expand/Conv2D/ReadVariableOpReadVariableOpblock_11_expand/kernel*
dtype0*'
_output_shapes
:`�
�
block_11_expand/Conv2DConv2D"block_10_project_BN/FusedBatchNorm%block_11_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
)block_11_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?*+
_class!
loc:@block_11_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_11_expand_BN/gammaVarHandleOp*
shape:�*)
shared_nameblock_11_expand_BN/gamma*+
_class!
loc:@block_11_expand_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_11_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_11_expand_BN/gamma*
_output_shapes
: 
�
block_11_expand_BN/gamma/AssignAssignVariableOpblock_11_expand_BN/gamma)block_11_expand_BN/gamma/Initializer/ones*+
_class!
loc:@block_11_expand_BN/gamma*
dtype0
�
,block_11_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_11_expand_BN/gamma*+
_class!
loc:@block_11_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
)block_11_expand_BN/beta/Initializer/zerosConst*
valueB�*    **
_class 
loc:@block_11_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_11_expand_BN/betaVarHandleOp*
shape:�*(
shared_nameblock_11_expand_BN/beta**
_class 
loc:@block_11_expand_BN/beta*
dtype0*
_output_shapes
: 

8block_11_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_11_expand_BN/beta*
_output_shapes
: 
�
block_11_expand_BN/beta/AssignAssignVariableOpblock_11_expand_BN/beta)block_11_expand_BN/beta/Initializer/zeros**
_class 
loc:@block_11_expand_BN/beta*
dtype0
�
+block_11_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_11_expand_BN/beta**
_class 
loc:@block_11_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
0block_11_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *1
_class'
%#loc:@block_11_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_11_expand_BN/moving_meanVarHandleOp*
shape:�*/
shared_name block_11_expand_BN/moving_mean*1
_class'
%#loc:@block_11_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_11_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_11_expand_BN/moving_mean*
_output_shapes
: 
�
%block_11_expand_BN/moving_mean/AssignAssignVariableOpblock_11_expand_BN/moving_mean0block_11_expand_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_11_expand_BN/moving_mean*
dtype0
�
2block_11_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_11_expand_BN/moving_mean*1
_class'
%#loc:@block_11_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
3block_11_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*5
_class+
)'loc:@block_11_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
"block_11_expand_BN/moving_varianceVarHandleOp*
shape:�*3
shared_name$"block_11_expand_BN/moving_variance*5
_class+
)'loc:@block_11_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_11_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_11_expand_BN/moving_variance*
_output_shapes
: 
�
)block_11_expand_BN/moving_variance/AssignAssignVariableOp"block_11_expand_BN/moving_variance3block_11_expand_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_11_expand_BN/moving_variance*
dtype0
�
6block_11_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_11_expand_BN/moving_variance*5
_class+
)'loc:@block_11_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
w
!block_11_expand_BN/ReadVariableOpReadVariableOpblock_11_expand_BN/gamma*
dtype0*
_output_shapes	
:�
x
#block_11_expand_BN/ReadVariableOp_1ReadVariableOpblock_11_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
0block_11_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_11_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_11_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_11_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_11_expand_BN/FusedBatchNormFusedBatchNormblock_11_expand/Conv2D!block_11_expand_BN/ReadVariableOp#block_11_expand_BN/ReadVariableOp_10block_11_expand_BN/FusedBatchNorm/ReadVariableOp2block_11_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
]
block_11_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_11_expand_relu/Relu6Relu6!block_11_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
Dblock_11_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      @     *6
_class,
*(loc:@block_11_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Bblock_11_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *[:�*6
_class,
*(loc:@block_11_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Bblock_11_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *[:=*6
_class,
*(loc:@block_11_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Lblock_11_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformDblock_11_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@block_11_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Bblock_11_depthwise/depthwise_kernel/Initializer/random_uniform/subSubBblock_11_depthwise/depthwise_kernel/Initializer/random_uniform/maxBblock_11_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@block_11_depthwise/depthwise_kernel*
_output_shapes
: 
�
Bblock_11_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulLblock_11_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformBblock_11_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@block_11_depthwise/depthwise_kernel*'
_output_shapes
:�
�
>block_11_depthwise/depthwise_kernel/Initializer/random_uniformAddBblock_11_depthwise/depthwise_kernel/Initializer/random_uniform/mulBblock_11_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@block_11_depthwise/depthwise_kernel*'
_output_shapes
:�
�
#block_11_depthwise/depthwise_kernelVarHandleOp*
shape:�*4
shared_name%#block_11_depthwise/depthwise_kernel*6
_class,
*(loc:@block_11_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Dblock_11_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp#block_11_depthwise/depthwise_kernel*
_output_shapes
: 
�
*block_11_depthwise/depthwise_kernel/AssignAssignVariableOp#block_11_depthwise/depthwise_kernel>block_11_depthwise/depthwise_kernel/Initializer/random_uniform*6
_class,
*(loc:@block_11_depthwise/depthwise_kernel*
dtype0
�
7block_11_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp#block_11_depthwise/depthwise_kernel*6
_class,
*(loc:@block_11_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
+block_11_depthwise/depthwise/ReadVariableOpReadVariableOp#block_11_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
{
"block_11_depthwise/depthwise/ShapeConst*%
valueB"      @     *
dtype0*
_output_shapes
:
{
*block_11_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_11_depthwise/depthwiseDepthwiseConv2dNativeblock_11_expand_relu/Relu6+block_11_depthwise/depthwise/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
,block_11_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*.
_class$
" loc:@block_11_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_11_depthwise_BN/gammaVarHandleOp*
shape:�*,
shared_nameblock_11_depthwise_BN/gamma*.
_class$
" loc:@block_11_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
<block_11_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_11_depthwise_BN/gamma*
_output_shapes
: 
�
"block_11_depthwise_BN/gamma/AssignAssignVariableOpblock_11_depthwise_BN/gamma,block_11_depthwise_BN/gamma/Initializer/ones*.
_class$
" loc:@block_11_depthwise_BN/gamma*
dtype0
�
/block_11_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_11_depthwise_BN/gamma*.
_class$
" loc:@block_11_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
,block_11_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *-
_class#
!loc:@block_11_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_11_depthwise_BN/betaVarHandleOp*
shape:�*+
shared_nameblock_11_depthwise_BN/beta*-
_class#
!loc:@block_11_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
;block_11_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_11_depthwise_BN/beta*
_output_shapes
: 
�
!block_11_depthwise_BN/beta/AssignAssignVariableOpblock_11_depthwise_BN/beta,block_11_depthwise_BN/beta/Initializer/zeros*-
_class#
!loc:@block_11_depthwise_BN/beta*
dtype0
�
.block_11_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_11_depthwise_BN/beta*-
_class#
!loc:@block_11_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
3block_11_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *4
_class*
(&loc:@block_11_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
!block_11_depthwise_BN/moving_meanVarHandleOp*
shape:�*2
shared_name#!block_11_depthwise_BN/moving_mean*4
_class*
(&loc:@block_11_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Bblock_11_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_11_depthwise_BN/moving_mean*
_output_shapes
: 
�
(block_11_depthwise_BN/moving_mean/AssignAssignVariableOp!block_11_depthwise_BN/moving_mean3block_11_depthwise_BN/moving_mean/Initializer/zeros*4
_class*
(&loc:@block_11_depthwise_BN/moving_mean*
dtype0
�
5block_11_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp!block_11_depthwise_BN/moving_mean*4
_class*
(&loc:@block_11_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
6block_11_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*8
_class.
,*loc:@block_11_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
%block_11_depthwise_BN/moving_varianceVarHandleOp*
shape:�*6
shared_name'%block_11_depthwise_BN/moving_variance*8
_class.
,*loc:@block_11_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Fblock_11_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%block_11_depthwise_BN/moving_variance*
_output_shapes
: 
�
,block_11_depthwise_BN/moving_variance/AssignAssignVariableOp%block_11_depthwise_BN/moving_variance6block_11_depthwise_BN/moving_variance/Initializer/ones*8
_class.
,*loc:@block_11_depthwise_BN/moving_variance*
dtype0
�
9block_11_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp%block_11_depthwise_BN/moving_variance*8
_class.
,*loc:@block_11_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
}
$block_11_depthwise_BN/ReadVariableOpReadVariableOpblock_11_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
~
&block_11_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_11_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
3block_11_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp!block_11_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_11_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp%block_11_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_11_depthwise_BN/FusedBatchNormFusedBatchNormblock_11_depthwise/depthwise$block_11_depthwise_BN/ReadVariableOp&block_11_depthwise_BN/ReadVariableOp_13block_11_depthwise_BN/FusedBatchNorm/ReadVariableOp5block_11_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
`
block_11_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_11_depthwise_relu/Relu6Relu6$block_11_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
8block_11_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @  `   **
_class 
loc:@block_11_project/kernel*
dtype0*
_output_shapes
:
�
6block_11_project/kernel/Initializer/random_uniform/minConst*
valueB
 *����**
_class 
loc:@block_11_project/kernel*
dtype0*
_output_shapes
: 
�
6block_11_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *���=**
_class 
loc:@block_11_project/kernel*
dtype0*
_output_shapes
: 
�
@block_11_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform8block_11_project/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@block_11_project/kernel*
dtype0*'
_output_shapes
:�`
�
6block_11_project/kernel/Initializer/random_uniform/subSub6block_11_project/kernel/Initializer/random_uniform/max6block_11_project/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@block_11_project/kernel*
_output_shapes
: 
�
6block_11_project/kernel/Initializer/random_uniform/mulMul@block_11_project/kernel/Initializer/random_uniform/RandomUniform6block_11_project/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@block_11_project/kernel*'
_output_shapes
:�`
�
2block_11_project/kernel/Initializer/random_uniformAdd6block_11_project/kernel/Initializer/random_uniform/mul6block_11_project/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@block_11_project/kernel*'
_output_shapes
:�`
�
block_11_project/kernelVarHandleOp*
shape:�`*(
shared_nameblock_11_project/kernel**
_class 
loc:@block_11_project/kernel*
dtype0*
_output_shapes
: 

8block_11_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_11_project/kernel*
_output_shapes
: 
�
block_11_project/kernel/AssignAssignVariableOpblock_11_project/kernel2block_11_project/kernel/Initializer/random_uniform**
_class 
loc:@block_11_project/kernel*
dtype0
�
+block_11_project/kernel/Read/ReadVariableOpReadVariableOpblock_11_project/kernel**
_class 
loc:@block_11_project/kernel*
dtype0*'
_output_shapes
:�`
o
block_11_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
&block_11_project/Conv2D/ReadVariableOpReadVariableOpblock_11_project/kernel*
dtype0*'
_output_shapes
:�`
�
block_11_project/Conv2DConv2Dblock_11_depthwise_relu/Relu6&block_11_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:���������`
�
*block_11_project_BN/gamma/Initializer/onesConst*
valueB`*  �?*,
_class"
 loc:@block_11_project_BN/gamma*
dtype0*
_output_shapes
:`
�
block_11_project_BN/gammaVarHandleOp*
shape:`**
shared_nameblock_11_project_BN/gamma*,
_class"
 loc:@block_11_project_BN/gamma*
dtype0*
_output_shapes
: 
�
:block_11_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_11_project_BN/gamma*
_output_shapes
: 
�
 block_11_project_BN/gamma/AssignAssignVariableOpblock_11_project_BN/gamma*block_11_project_BN/gamma/Initializer/ones*,
_class"
 loc:@block_11_project_BN/gamma*
dtype0
�
-block_11_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_11_project_BN/gamma*,
_class"
 loc:@block_11_project_BN/gamma*
dtype0*
_output_shapes
:`
�
*block_11_project_BN/beta/Initializer/zerosConst*
valueB`*    *+
_class!
loc:@block_11_project_BN/beta*
dtype0*
_output_shapes
:`
�
block_11_project_BN/betaVarHandleOp*
shape:`*)
shared_nameblock_11_project_BN/beta*+
_class!
loc:@block_11_project_BN/beta*
dtype0*
_output_shapes
: 
�
9block_11_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_11_project_BN/beta*
_output_shapes
: 
�
block_11_project_BN/beta/AssignAssignVariableOpblock_11_project_BN/beta*block_11_project_BN/beta/Initializer/zeros*+
_class!
loc:@block_11_project_BN/beta*
dtype0
�
,block_11_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_11_project_BN/beta*+
_class!
loc:@block_11_project_BN/beta*
dtype0*
_output_shapes
:`
�
1block_11_project_BN/moving_mean/Initializer/zerosConst*
valueB`*    *2
_class(
&$loc:@block_11_project_BN/moving_mean*
dtype0*
_output_shapes
:`
�
block_11_project_BN/moving_meanVarHandleOp*
shape:`*0
shared_name!block_11_project_BN/moving_mean*2
_class(
&$loc:@block_11_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
@block_11_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_11_project_BN/moving_mean*
_output_shapes
: 
�
&block_11_project_BN/moving_mean/AssignAssignVariableOpblock_11_project_BN/moving_mean1block_11_project_BN/moving_mean/Initializer/zeros*2
_class(
&$loc:@block_11_project_BN/moving_mean*
dtype0
�
3block_11_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_11_project_BN/moving_mean*2
_class(
&$loc:@block_11_project_BN/moving_mean*
dtype0*
_output_shapes
:`
�
4block_11_project_BN/moving_variance/Initializer/onesConst*
valueB`*  �?*6
_class,
*(loc:@block_11_project_BN/moving_variance*
dtype0*
_output_shapes
:`
�
#block_11_project_BN/moving_varianceVarHandleOp*
shape:`*4
shared_name%#block_11_project_BN/moving_variance*6
_class,
*(loc:@block_11_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Dblock_11_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#block_11_project_BN/moving_variance*
_output_shapes
: 
�
*block_11_project_BN/moving_variance/AssignAssignVariableOp#block_11_project_BN/moving_variance4block_11_project_BN/moving_variance/Initializer/ones*6
_class,
*(loc:@block_11_project_BN/moving_variance*
dtype0
�
7block_11_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp#block_11_project_BN/moving_variance*6
_class,
*(loc:@block_11_project_BN/moving_variance*
dtype0*
_output_shapes
:`
x
"block_11_project_BN/ReadVariableOpReadVariableOpblock_11_project_BN/gamma*
dtype0*
_output_shapes
:`
y
$block_11_project_BN/ReadVariableOp_1ReadVariableOpblock_11_project_BN/beta*
dtype0*
_output_shapes
:`
�
1block_11_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_11_project_BN/moving_mean*
dtype0*
_output_shapes
:`
�
3block_11_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp#block_11_project_BN/moving_variance*
dtype0*
_output_shapes
:`
�
"block_11_project_BN/FusedBatchNormFusedBatchNormblock_11_project/Conv2D"block_11_project_BN/ReadVariableOp$block_11_project_BN/ReadVariableOp_11block_11_project_BN/FusedBatchNorm/ReadVariableOp3block_11_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:���������`:`:`:`:`*
is_training( 
^
block_11_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_11_add/addAdd"block_10_project_BN/FusedBatchNorm"block_11_project_BN/FusedBatchNorm*
T0*/
_output_shapes
:���������`
�
7block_12_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"      `   @  *)
_class
loc:@block_12_expand/kernel*
dtype0*
_output_shapes
:
�
5block_12_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *����*)
_class
loc:@block_12_expand/kernel*
dtype0*
_output_shapes
: 
�
5block_12_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *���=*)
_class
loc:@block_12_expand/kernel*
dtype0*
_output_shapes
: 
�
?block_12_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_12_expand/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_12_expand/kernel*
dtype0*'
_output_shapes
:`�
�
5block_12_expand/kernel/Initializer/random_uniform/subSub5block_12_expand/kernel/Initializer/random_uniform/max5block_12_expand/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_12_expand/kernel*
_output_shapes
: 
�
5block_12_expand/kernel/Initializer/random_uniform/mulMul?block_12_expand/kernel/Initializer/random_uniform/RandomUniform5block_12_expand/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_12_expand/kernel*'
_output_shapes
:`�
�
1block_12_expand/kernel/Initializer/random_uniformAdd5block_12_expand/kernel/Initializer/random_uniform/mul5block_12_expand/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_12_expand/kernel*'
_output_shapes
:`�
�
block_12_expand/kernelVarHandleOp*
shape:`�*'
shared_nameblock_12_expand/kernel*)
_class
loc:@block_12_expand/kernel*
dtype0*
_output_shapes
: 
}
7block_12_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_12_expand/kernel*
_output_shapes
: 
�
block_12_expand/kernel/AssignAssignVariableOpblock_12_expand/kernel1block_12_expand/kernel/Initializer/random_uniform*)
_class
loc:@block_12_expand/kernel*
dtype0
�
*block_12_expand/kernel/Read/ReadVariableOpReadVariableOpblock_12_expand/kernel*)
_class
loc:@block_12_expand/kernel*
dtype0*'
_output_shapes
:`�
n
block_12_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_12_expand/Conv2D/ReadVariableOpReadVariableOpblock_12_expand/kernel*
dtype0*'
_output_shapes
:`�
�
block_12_expand/Conv2DConv2Dblock_11_add/add%block_12_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
)block_12_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?*+
_class!
loc:@block_12_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_12_expand_BN/gammaVarHandleOp*
shape:�*)
shared_nameblock_12_expand_BN/gamma*+
_class!
loc:@block_12_expand_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_12_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_12_expand_BN/gamma*
_output_shapes
: 
�
block_12_expand_BN/gamma/AssignAssignVariableOpblock_12_expand_BN/gamma)block_12_expand_BN/gamma/Initializer/ones*+
_class!
loc:@block_12_expand_BN/gamma*
dtype0
�
,block_12_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_12_expand_BN/gamma*+
_class!
loc:@block_12_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
)block_12_expand_BN/beta/Initializer/zerosConst*
valueB�*    **
_class 
loc:@block_12_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_12_expand_BN/betaVarHandleOp*
shape:�*(
shared_nameblock_12_expand_BN/beta**
_class 
loc:@block_12_expand_BN/beta*
dtype0*
_output_shapes
: 

8block_12_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_12_expand_BN/beta*
_output_shapes
: 
�
block_12_expand_BN/beta/AssignAssignVariableOpblock_12_expand_BN/beta)block_12_expand_BN/beta/Initializer/zeros**
_class 
loc:@block_12_expand_BN/beta*
dtype0
�
+block_12_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_12_expand_BN/beta**
_class 
loc:@block_12_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
0block_12_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *1
_class'
%#loc:@block_12_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_12_expand_BN/moving_meanVarHandleOp*
shape:�*/
shared_name block_12_expand_BN/moving_mean*1
_class'
%#loc:@block_12_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_12_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_12_expand_BN/moving_mean*
_output_shapes
: 
�
%block_12_expand_BN/moving_mean/AssignAssignVariableOpblock_12_expand_BN/moving_mean0block_12_expand_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_12_expand_BN/moving_mean*
dtype0
�
2block_12_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_12_expand_BN/moving_mean*1
_class'
%#loc:@block_12_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
3block_12_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*5
_class+
)'loc:@block_12_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
"block_12_expand_BN/moving_varianceVarHandleOp*
shape:�*3
shared_name$"block_12_expand_BN/moving_variance*5
_class+
)'loc:@block_12_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_12_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_12_expand_BN/moving_variance*
_output_shapes
: 
�
)block_12_expand_BN/moving_variance/AssignAssignVariableOp"block_12_expand_BN/moving_variance3block_12_expand_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_12_expand_BN/moving_variance*
dtype0
�
6block_12_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_12_expand_BN/moving_variance*5
_class+
)'loc:@block_12_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
w
!block_12_expand_BN/ReadVariableOpReadVariableOpblock_12_expand_BN/gamma*
dtype0*
_output_shapes	
:�
x
#block_12_expand_BN/ReadVariableOp_1ReadVariableOpblock_12_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
0block_12_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_12_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_12_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_12_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_12_expand_BN/FusedBatchNormFusedBatchNormblock_12_expand/Conv2D!block_12_expand_BN/ReadVariableOp#block_12_expand_BN/ReadVariableOp_10block_12_expand_BN/FusedBatchNorm/ReadVariableOp2block_12_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
]
block_12_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_12_expand_relu/Relu6Relu6!block_12_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
Dblock_12_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      @     *6
_class,
*(loc:@block_12_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Bblock_12_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *[:�*6
_class,
*(loc:@block_12_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Bblock_12_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *[:=*6
_class,
*(loc:@block_12_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Lblock_12_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformDblock_12_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@block_12_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Bblock_12_depthwise/depthwise_kernel/Initializer/random_uniform/subSubBblock_12_depthwise/depthwise_kernel/Initializer/random_uniform/maxBblock_12_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@block_12_depthwise/depthwise_kernel*
_output_shapes
: 
�
Bblock_12_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulLblock_12_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformBblock_12_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@block_12_depthwise/depthwise_kernel*'
_output_shapes
:�
�
>block_12_depthwise/depthwise_kernel/Initializer/random_uniformAddBblock_12_depthwise/depthwise_kernel/Initializer/random_uniform/mulBblock_12_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@block_12_depthwise/depthwise_kernel*'
_output_shapes
:�
�
#block_12_depthwise/depthwise_kernelVarHandleOp*
shape:�*4
shared_name%#block_12_depthwise/depthwise_kernel*6
_class,
*(loc:@block_12_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Dblock_12_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp#block_12_depthwise/depthwise_kernel*
_output_shapes
: 
�
*block_12_depthwise/depthwise_kernel/AssignAssignVariableOp#block_12_depthwise/depthwise_kernel>block_12_depthwise/depthwise_kernel/Initializer/random_uniform*6
_class,
*(loc:@block_12_depthwise/depthwise_kernel*
dtype0
�
7block_12_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp#block_12_depthwise/depthwise_kernel*6
_class,
*(loc:@block_12_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
+block_12_depthwise/depthwise/ReadVariableOpReadVariableOp#block_12_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
{
"block_12_depthwise/depthwise/ShapeConst*%
valueB"      @     *
dtype0*
_output_shapes
:
{
*block_12_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_12_depthwise/depthwiseDepthwiseConv2dNativeblock_12_expand_relu/Relu6+block_12_depthwise/depthwise/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
,block_12_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*.
_class$
" loc:@block_12_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_12_depthwise_BN/gammaVarHandleOp*
shape:�*,
shared_nameblock_12_depthwise_BN/gamma*.
_class$
" loc:@block_12_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
<block_12_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_12_depthwise_BN/gamma*
_output_shapes
: 
�
"block_12_depthwise_BN/gamma/AssignAssignVariableOpblock_12_depthwise_BN/gamma,block_12_depthwise_BN/gamma/Initializer/ones*.
_class$
" loc:@block_12_depthwise_BN/gamma*
dtype0
�
/block_12_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_12_depthwise_BN/gamma*.
_class$
" loc:@block_12_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
,block_12_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *-
_class#
!loc:@block_12_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_12_depthwise_BN/betaVarHandleOp*
shape:�*+
shared_nameblock_12_depthwise_BN/beta*-
_class#
!loc:@block_12_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
;block_12_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_12_depthwise_BN/beta*
_output_shapes
: 
�
!block_12_depthwise_BN/beta/AssignAssignVariableOpblock_12_depthwise_BN/beta,block_12_depthwise_BN/beta/Initializer/zeros*-
_class#
!loc:@block_12_depthwise_BN/beta*
dtype0
�
.block_12_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_12_depthwise_BN/beta*-
_class#
!loc:@block_12_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
3block_12_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *4
_class*
(&loc:@block_12_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
!block_12_depthwise_BN/moving_meanVarHandleOp*
shape:�*2
shared_name#!block_12_depthwise_BN/moving_mean*4
_class*
(&loc:@block_12_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Bblock_12_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_12_depthwise_BN/moving_mean*
_output_shapes
: 
�
(block_12_depthwise_BN/moving_mean/AssignAssignVariableOp!block_12_depthwise_BN/moving_mean3block_12_depthwise_BN/moving_mean/Initializer/zeros*4
_class*
(&loc:@block_12_depthwise_BN/moving_mean*
dtype0
�
5block_12_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp!block_12_depthwise_BN/moving_mean*4
_class*
(&loc:@block_12_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
6block_12_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*8
_class.
,*loc:@block_12_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
%block_12_depthwise_BN/moving_varianceVarHandleOp*
shape:�*6
shared_name'%block_12_depthwise_BN/moving_variance*8
_class.
,*loc:@block_12_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Fblock_12_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%block_12_depthwise_BN/moving_variance*
_output_shapes
: 
�
,block_12_depthwise_BN/moving_variance/AssignAssignVariableOp%block_12_depthwise_BN/moving_variance6block_12_depthwise_BN/moving_variance/Initializer/ones*8
_class.
,*loc:@block_12_depthwise_BN/moving_variance*
dtype0
�
9block_12_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp%block_12_depthwise_BN/moving_variance*8
_class.
,*loc:@block_12_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
}
$block_12_depthwise_BN/ReadVariableOpReadVariableOpblock_12_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
~
&block_12_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_12_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
3block_12_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp!block_12_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_12_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp%block_12_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_12_depthwise_BN/FusedBatchNormFusedBatchNormblock_12_depthwise/depthwise$block_12_depthwise_BN/ReadVariableOp&block_12_depthwise_BN/ReadVariableOp_13block_12_depthwise_BN/FusedBatchNorm/ReadVariableOp5block_12_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
`
block_12_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_12_depthwise_relu/Relu6Relu6$block_12_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
8block_12_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @  `   **
_class 
loc:@block_12_project/kernel*
dtype0*
_output_shapes
:
�
6block_12_project/kernel/Initializer/random_uniform/minConst*
valueB
 *����**
_class 
loc:@block_12_project/kernel*
dtype0*
_output_shapes
: 
�
6block_12_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *���=**
_class 
loc:@block_12_project/kernel*
dtype0*
_output_shapes
: 
�
@block_12_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform8block_12_project/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@block_12_project/kernel*
dtype0*'
_output_shapes
:�`
�
6block_12_project/kernel/Initializer/random_uniform/subSub6block_12_project/kernel/Initializer/random_uniform/max6block_12_project/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@block_12_project/kernel*
_output_shapes
: 
�
6block_12_project/kernel/Initializer/random_uniform/mulMul@block_12_project/kernel/Initializer/random_uniform/RandomUniform6block_12_project/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@block_12_project/kernel*'
_output_shapes
:�`
�
2block_12_project/kernel/Initializer/random_uniformAdd6block_12_project/kernel/Initializer/random_uniform/mul6block_12_project/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@block_12_project/kernel*'
_output_shapes
:�`
�
block_12_project/kernelVarHandleOp*
shape:�`*(
shared_nameblock_12_project/kernel**
_class 
loc:@block_12_project/kernel*
dtype0*
_output_shapes
: 

8block_12_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_12_project/kernel*
_output_shapes
: 
�
block_12_project/kernel/AssignAssignVariableOpblock_12_project/kernel2block_12_project/kernel/Initializer/random_uniform**
_class 
loc:@block_12_project/kernel*
dtype0
�
+block_12_project/kernel/Read/ReadVariableOpReadVariableOpblock_12_project/kernel**
_class 
loc:@block_12_project/kernel*
dtype0*'
_output_shapes
:�`
o
block_12_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
&block_12_project/Conv2D/ReadVariableOpReadVariableOpblock_12_project/kernel*
dtype0*'
_output_shapes
:�`
�
block_12_project/Conv2DConv2Dblock_12_depthwise_relu/Relu6&block_12_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*/
_output_shapes
:���������`
�
*block_12_project_BN/gamma/Initializer/onesConst*
valueB`*  �?*,
_class"
 loc:@block_12_project_BN/gamma*
dtype0*
_output_shapes
:`
�
block_12_project_BN/gammaVarHandleOp*
shape:`**
shared_nameblock_12_project_BN/gamma*,
_class"
 loc:@block_12_project_BN/gamma*
dtype0*
_output_shapes
: 
�
:block_12_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_12_project_BN/gamma*
_output_shapes
: 
�
 block_12_project_BN/gamma/AssignAssignVariableOpblock_12_project_BN/gamma*block_12_project_BN/gamma/Initializer/ones*,
_class"
 loc:@block_12_project_BN/gamma*
dtype0
�
-block_12_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_12_project_BN/gamma*,
_class"
 loc:@block_12_project_BN/gamma*
dtype0*
_output_shapes
:`
�
*block_12_project_BN/beta/Initializer/zerosConst*
valueB`*    *+
_class!
loc:@block_12_project_BN/beta*
dtype0*
_output_shapes
:`
�
block_12_project_BN/betaVarHandleOp*
shape:`*)
shared_nameblock_12_project_BN/beta*+
_class!
loc:@block_12_project_BN/beta*
dtype0*
_output_shapes
: 
�
9block_12_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_12_project_BN/beta*
_output_shapes
: 
�
block_12_project_BN/beta/AssignAssignVariableOpblock_12_project_BN/beta*block_12_project_BN/beta/Initializer/zeros*+
_class!
loc:@block_12_project_BN/beta*
dtype0
�
,block_12_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_12_project_BN/beta*+
_class!
loc:@block_12_project_BN/beta*
dtype0*
_output_shapes
:`
�
1block_12_project_BN/moving_mean/Initializer/zerosConst*
valueB`*    *2
_class(
&$loc:@block_12_project_BN/moving_mean*
dtype0*
_output_shapes
:`
�
block_12_project_BN/moving_meanVarHandleOp*
shape:`*0
shared_name!block_12_project_BN/moving_mean*2
_class(
&$loc:@block_12_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
@block_12_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_12_project_BN/moving_mean*
_output_shapes
: 
�
&block_12_project_BN/moving_mean/AssignAssignVariableOpblock_12_project_BN/moving_mean1block_12_project_BN/moving_mean/Initializer/zeros*2
_class(
&$loc:@block_12_project_BN/moving_mean*
dtype0
�
3block_12_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_12_project_BN/moving_mean*2
_class(
&$loc:@block_12_project_BN/moving_mean*
dtype0*
_output_shapes
:`
�
4block_12_project_BN/moving_variance/Initializer/onesConst*
valueB`*  �?*6
_class,
*(loc:@block_12_project_BN/moving_variance*
dtype0*
_output_shapes
:`
�
#block_12_project_BN/moving_varianceVarHandleOp*
shape:`*4
shared_name%#block_12_project_BN/moving_variance*6
_class,
*(loc:@block_12_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Dblock_12_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#block_12_project_BN/moving_variance*
_output_shapes
: 
�
*block_12_project_BN/moving_variance/AssignAssignVariableOp#block_12_project_BN/moving_variance4block_12_project_BN/moving_variance/Initializer/ones*6
_class,
*(loc:@block_12_project_BN/moving_variance*
dtype0
�
7block_12_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp#block_12_project_BN/moving_variance*6
_class,
*(loc:@block_12_project_BN/moving_variance*
dtype0*
_output_shapes
:`
x
"block_12_project_BN/ReadVariableOpReadVariableOpblock_12_project_BN/gamma*
dtype0*
_output_shapes
:`
y
$block_12_project_BN/ReadVariableOp_1ReadVariableOpblock_12_project_BN/beta*
dtype0*
_output_shapes
:`
�
1block_12_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_12_project_BN/moving_mean*
dtype0*
_output_shapes
:`
�
3block_12_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp#block_12_project_BN/moving_variance*
dtype0*
_output_shapes
:`
�
"block_12_project_BN/FusedBatchNormFusedBatchNormblock_12_project/Conv2D"block_12_project_BN/ReadVariableOp$block_12_project_BN/ReadVariableOp_11block_12_project_BN/FusedBatchNorm/ReadVariableOp3block_12_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*G
_output_shapes5
3:���������`:`:`:`:`*
is_training( 
^
block_12_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_12_add/addAddblock_11_add/add"block_12_project_BN/FusedBatchNorm*
T0*/
_output_shapes
:���������`
�
7block_13_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"      `   @  *)
_class
loc:@block_13_expand/kernel*
dtype0*
_output_shapes
:
�
5block_13_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *����*)
_class
loc:@block_13_expand/kernel*
dtype0*
_output_shapes
: 
�
5block_13_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *���=*)
_class
loc:@block_13_expand/kernel*
dtype0*
_output_shapes
: 
�
?block_13_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_13_expand/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_13_expand/kernel*
dtype0*'
_output_shapes
:`�
�
5block_13_expand/kernel/Initializer/random_uniform/subSub5block_13_expand/kernel/Initializer/random_uniform/max5block_13_expand/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_13_expand/kernel*
_output_shapes
: 
�
5block_13_expand/kernel/Initializer/random_uniform/mulMul?block_13_expand/kernel/Initializer/random_uniform/RandomUniform5block_13_expand/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_13_expand/kernel*'
_output_shapes
:`�
�
1block_13_expand/kernel/Initializer/random_uniformAdd5block_13_expand/kernel/Initializer/random_uniform/mul5block_13_expand/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_13_expand/kernel*'
_output_shapes
:`�
�
block_13_expand/kernelVarHandleOp*
shape:`�*'
shared_nameblock_13_expand/kernel*)
_class
loc:@block_13_expand/kernel*
dtype0*
_output_shapes
: 
}
7block_13_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_13_expand/kernel*
_output_shapes
: 
�
block_13_expand/kernel/AssignAssignVariableOpblock_13_expand/kernel1block_13_expand/kernel/Initializer/random_uniform*)
_class
loc:@block_13_expand/kernel*
dtype0
�
*block_13_expand/kernel/Read/ReadVariableOpReadVariableOpblock_13_expand/kernel*)
_class
loc:@block_13_expand/kernel*
dtype0*'
_output_shapes
:`�
n
block_13_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_13_expand/Conv2D/ReadVariableOpReadVariableOpblock_13_expand/kernel*
dtype0*'
_output_shapes
:`�
�
block_13_expand/Conv2DConv2Dblock_12_add/add%block_13_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
)block_13_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?*+
_class!
loc:@block_13_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_13_expand_BN/gammaVarHandleOp*
shape:�*)
shared_nameblock_13_expand_BN/gamma*+
_class!
loc:@block_13_expand_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_13_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_13_expand_BN/gamma*
_output_shapes
: 
�
block_13_expand_BN/gamma/AssignAssignVariableOpblock_13_expand_BN/gamma)block_13_expand_BN/gamma/Initializer/ones*+
_class!
loc:@block_13_expand_BN/gamma*
dtype0
�
,block_13_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_13_expand_BN/gamma*+
_class!
loc:@block_13_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
)block_13_expand_BN/beta/Initializer/zerosConst*
valueB�*    **
_class 
loc:@block_13_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_13_expand_BN/betaVarHandleOp*
shape:�*(
shared_nameblock_13_expand_BN/beta**
_class 
loc:@block_13_expand_BN/beta*
dtype0*
_output_shapes
: 

8block_13_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_13_expand_BN/beta*
_output_shapes
: 
�
block_13_expand_BN/beta/AssignAssignVariableOpblock_13_expand_BN/beta)block_13_expand_BN/beta/Initializer/zeros**
_class 
loc:@block_13_expand_BN/beta*
dtype0
�
+block_13_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_13_expand_BN/beta**
_class 
loc:@block_13_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
0block_13_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *1
_class'
%#loc:@block_13_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_13_expand_BN/moving_meanVarHandleOp*
shape:�*/
shared_name block_13_expand_BN/moving_mean*1
_class'
%#loc:@block_13_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_13_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_13_expand_BN/moving_mean*
_output_shapes
: 
�
%block_13_expand_BN/moving_mean/AssignAssignVariableOpblock_13_expand_BN/moving_mean0block_13_expand_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_13_expand_BN/moving_mean*
dtype0
�
2block_13_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_13_expand_BN/moving_mean*1
_class'
%#loc:@block_13_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
3block_13_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*5
_class+
)'loc:@block_13_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
"block_13_expand_BN/moving_varianceVarHandleOp*
shape:�*3
shared_name$"block_13_expand_BN/moving_variance*5
_class+
)'loc:@block_13_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_13_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_13_expand_BN/moving_variance*
_output_shapes
: 
�
)block_13_expand_BN/moving_variance/AssignAssignVariableOp"block_13_expand_BN/moving_variance3block_13_expand_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_13_expand_BN/moving_variance*
dtype0
�
6block_13_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_13_expand_BN/moving_variance*5
_class+
)'loc:@block_13_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
w
!block_13_expand_BN/ReadVariableOpReadVariableOpblock_13_expand_BN/gamma*
dtype0*
_output_shapes	
:�
x
#block_13_expand_BN/ReadVariableOp_1ReadVariableOpblock_13_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
0block_13_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_13_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_13_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_13_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_13_expand_BN/FusedBatchNormFusedBatchNormblock_13_expand/Conv2D!block_13_expand_BN/ReadVariableOp#block_13_expand_BN/ReadVariableOp_10block_13_expand_BN/FusedBatchNorm/ReadVariableOp2block_13_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
]
block_13_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_13_expand_relu/Relu6Relu6!block_13_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
block_13_pad/Pad/paddingsConst*9
value0B."                               *
dtype0*
_output_shapes

:
�
block_13_pad/PadPadblock_13_expand_relu/Relu6block_13_pad/Pad/paddings*
T0*0
_output_shapes
:����������
�
Dblock_13_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      @     *6
_class,
*(loc:@block_13_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Bblock_13_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *[:�*6
_class,
*(loc:@block_13_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Bblock_13_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *[:=*6
_class,
*(loc:@block_13_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Lblock_13_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformDblock_13_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@block_13_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Bblock_13_depthwise/depthwise_kernel/Initializer/random_uniform/subSubBblock_13_depthwise/depthwise_kernel/Initializer/random_uniform/maxBblock_13_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@block_13_depthwise/depthwise_kernel*
_output_shapes
: 
�
Bblock_13_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulLblock_13_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformBblock_13_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@block_13_depthwise/depthwise_kernel*'
_output_shapes
:�
�
>block_13_depthwise/depthwise_kernel/Initializer/random_uniformAddBblock_13_depthwise/depthwise_kernel/Initializer/random_uniform/mulBblock_13_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@block_13_depthwise/depthwise_kernel*'
_output_shapes
:�
�
#block_13_depthwise/depthwise_kernelVarHandleOp*
shape:�*4
shared_name%#block_13_depthwise/depthwise_kernel*6
_class,
*(loc:@block_13_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Dblock_13_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp#block_13_depthwise/depthwise_kernel*
_output_shapes
: 
�
*block_13_depthwise/depthwise_kernel/AssignAssignVariableOp#block_13_depthwise/depthwise_kernel>block_13_depthwise/depthwise_kernel/Initializer/random_uniform*6
_class,
*(loc:@block_13_depthwise/depthwise_kernel*
dtype0
�
7block_13_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp#block_13_depthwise/depthwise_kernel*6
_class,
*(loc:@block_13_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
+block_13_depthwise/depthwise/ReadVariableOpReadVariableOp#block_13_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
{
"block_13_depthwise/depthwise/ShapeConst*%
valueB"      @     *
dtype0*
_output_shapes
:
{
*block_13_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_13_depthwise/depthwiseDepthwiseConv2dNativeblock_13_pad/Pad+block_13_depthwise/depthwise/ReadVariableOp*
paddingVALID*
T0*
strides
*0
_output_shapes
:����������
�
,block_13_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*.
_class$
" loc:@block_13_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_13_depthwise_BN/gammaVarHandleOp*
shape:�*,
shared_nameblock_13_depthwise_BN/gamma*.
_class$
" loc:@block_13_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
<block_13_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_13_depthwise_BN/gamma*
_output_shapes
: 
�
"block_13_depthwise_BN/gamma/AssignAssignVariableOpblock_13_depthwise_BN/gamma,block_13_depthwise_BN/gamma/Initializer/ones*.
_class$
" loc:@block_13_depthwise_BN/gamma*
dtype0
�
/block_13_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_13_depthwise_BN/gamma*.
_class$
" loc:@block_13_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
,block_13_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *-
_class#
!loc:@block_13_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_13_depthwise_BN/betaVarHandleOp*
shape:�*+
shared_nameblock_13_depthwise_BN/beta*-
_class#
!loc:@block_13_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
;block_13_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_13_depthwise_BN/beta*
_output_shapes
: 
�
!block_13_depthwise_BN/beta/AssignAssignVariableOpblock_13_depthwise_BN/beta,block_13_depthwise_BN/beta/Initializer/zeros*-
_class#
!loc:@block_13_depthwise_BN/beta*
dtype0
�
.block_13_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_13_depthwise_BN/beta*-
_class#
!loc:@block_13_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
3block_13_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *4
_class*
(&loc:@block_13_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
!block_13_depthwise_BN/moving_meanVarHandleOp*
shape:�*2
shared_name#!block_13_depthwise_BN/moving_mean*4
_class*
(&loc:@block_13_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Bblock_13_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_13_depthwise_BN/moving_mean*
_output_shapes
: 
�
(block_13_depthwise_BN/moving_mean/AssignAssignVariableOp!block_13_depthwise_BN/moving_mean3block_13_depthwise_BN/moving_mean/Initializer/zeros*4
_class*
(&loc:@block_13_depthwise_BN/moving_mean*
dtype0
�
5block_13_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp!block_13_depthwise_BN/moving_mean*4
_class*
(&loc:@block_13_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
6block_13_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*8
_class.
,*loc:@block_13_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
%block_13_depthwise_BN/moving_varianceVarHandleOp*
shape:�*6
shared_name'%block_13_depthwise_BN/moving_variance*8
_class.
,*loc:@block_13_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Fblock_13_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%block_13_depthwise_BN/moving_variance*
_output_shapes
: 
�
,block_13_depthwise_BN/moving_variance/AssignAssignVariableOp%block_13_depthwise_BN/moving_variance6block_13_depthwise_BN/moving_variance/Initializer/ones*8
_class.
,*loc:@block_13_depthwise_BN/moving_variance*
dtype0
�
9block_13_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp%block_13_depthwise_BN/moving_variance*8
_class.
,*loc:@block_13_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
}
$block_13_depthwise_BN/ReadVariableOpReadVariableOpblock_13_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
~
&block_13_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_13_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
3block_13_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp!block_13_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_13_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp%block_13_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_13_depthwise_BN/FusedBatchNormFusedBatchNormblock_13_depthwise/depthwise$block_13_depthwise_BN/ReadVariableOp&block_13_depthwise_BN/ReadVariableOp_13block_13_depthwise_BN/FusedBatchNorm/ReadVariableOp5block_13_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
`
block_13_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_13_depthwise_relu/Relu6Relu6$block_13_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
8block_13_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @  �   **
_class 
loc:@block_13_project/kernel*
dtype0*
_output_shapes
:
�
6block_13_project/kernel/Initializer/random_uniform/minConst*
valueB
 *�鸽**
_class 
loc:@block_13_project/kernel*
dtype0*
_output_shapes
: 
�
6block_13_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=**
_class 
loc:@block_13_project/kernel*
dtype0*
_output_shapes
: 
�
@block_13_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform8block_13_project/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@block_13_project/kernel*
dtype0*(
_output_shapes
:��
�
6block_13_project/kernel/Initializer/random_uniform/subSub6block_13_project/kernel/Initializer/random_uniform/max6block_13_project/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@block_13_project/kernel*
_output_shapes
: 
�
6block_13_project/kernel/Initializer/random_uniform/mulMul@block_13_project/kernel/Initializer/random_uniform/RandomUniform6block_13_project/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@block_13_project/kernel*(
_output_shapes
:��
�
2block_13_project/kernel/Initializer/random_uniformAdd6block_13_project/kernel/Initializer/random_uniform/mul6block_13_project/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@block_13_project/kernel*(
_output_shapes
:��
�
block_13_project/kernelVarHandleOp*
shape:��*(
shared_nameblock_13_project/kernel**
_class 
loc:@block_13_project/kernel*
dtype0*
_output_shapes
: 

8block_13_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_13_project/kernel*
_output_shapes
: 
�
block_13_project/kernel/AssignAssignVariableOpblock_13_project/kernel2block_13_project/kernel/Initializer/random_uniform**
_class 
loc:@block_13_project/kernel*
dtype0
�
+block_13_project/kernel/Read/ReadVariableOpReadVariableOpblock_13_project/kernel**
_class 
loc:@block_13_project/kernel*
dtype0*(
_output_shapes
:��
o
block_13_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
&block_13_project/Conv2D/ReadVariableOpReadVariableOpblock_13_project/kernel*
dtype0*(
_output_shapes
:��
�
block_13_project/Conv2DConv2Dblock_13_depthwise_relu/Relu6&block_13_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
*block_13_project_BN/gamma/Initializer/onesConst*
valueB�*  �?*,
_class"
 loc:@block_13_project_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_13_project_BN/gammaVarHandleOp*
shape:�**
shared_nameblock_13_project_BN/gamma*,
_class"
 loc:@block_13_project_BN/gamma*
dtype0*
_output_shapes
: 
�
:block_13_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_13_project_BN/gamma*
_output_shapes
: 
�
 block_13_project_BN/gamma/AssignAssignVariableOpblock_13_project_BN/gamma*block_13_project_BN/gamma/Initializer/ones*,
_class"
 loc:@block_13_project_BN/gamma*
dtype0
�
-block_13_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_13_project_BN/gamma*,
_class"
 loc:@block_13_project_BN/gamma*
dtype0*
_output_shapes	
:�
�
*block_13_project_BN/beta/Initializer/zerosConst*
valueB�*    *+
_class!
loc:@block_13_project_BN/beta*
dtype0*
_output_shapes	
:�
�
block_13_project_BN/betaVarHandleOp*
shape:�*)
shared_nameblock_13_project_BN/beta*+
_class!
loc:@block_13_project_BN/beta*
dtype0*
_output_shapes
: 
�
9block_13_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_13_project_BN/beta*
_output_shapes
: 
�
block_13_project_BN/beta/AssignAssignVariableOpblock_13_project_BN/beta*block_13_project_BN/beta/Initializer/zeros*+
_class!
loc:@block_13_project_BN/beta*
dtype0
�
,block_13_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_13_project_BN/beta*+
_class!
loc:@block_13_project_BN/beta*
dtype0*
_output_shapes	
:�
�
1block_13_project_BN/moving_mean/Initializer/zerosConst*
valueB�*    *2
_class(
&$loc:@block_13_project_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_13_project_BN/moving_meanVarHandleOp*
shape:�*0
shared_name!block_13_project_BN/moving_mean*2
_class(
&$loc:@block_13_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
@block_13_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_13_project_BN/moving_mean*
_output_shapes
: 
�
&block_13_project_BN/moving_mean/AssignAssignVariableOpblock_13_project_BN/moving_mean1block_13_project_BN/moving_mean/Initializer/zeros*2
_class(
&$loc:@block_13_project_BN/moving_mean*
dtype0
�
3block_13_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_13_project_BN/moving_mean*2
_class(
&$loc:@block_13_project_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
4block_13_project_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*6
_class,
*(loc:@block_13_project_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
#block_13_project_BN/moving_varianceVarHandleOp*
shape:�*4
shared_name%#block_13_project_BN/moving_variance*6
_class,
*(loc:@block_13_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Dblock_13_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#block_13_project_BN/moving_variance*
_output_shapes
: 
�
*block_13_project_BN/moving_variance/AssignAssignVariableOp#block_13_project_BN/moving_variance4block_13_project_BN/moving_variance/Initializer/ones*6
_class,
*(loc:@block_13_project_BN/moving_variance*
dtype0
�
7block_13_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp#block_13_project_BN/moving_variance*6
_class,
*(loc:@block_13_project_BN/moving_variance*
dtype0*
_output_shapes	
:�
y
"block_13_project_BN/ReadVariableOpReadVariableOpblock_13_project_BN/gamma*
dtype0*
_output_shapes	
:�
z
$block_13_project_BN/ReadVariableOp_1ReadVariableOpblock_13_project_BN/beta*
dtype0*
_output_shapes	
:�
�
1block_13_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_13_project_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
3block_13_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp#block_13_project_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
"block_13_project_BN/FusedBatchNormFusedBatchNormblock_13_project/Conv2D"block_13_project_BN/ReadVariableOp$block_13_project_BN/ReadVariableOp_11block_13_project_BN/FusedBatchNorm/ReadVariableOp3block_13_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
^
block_13_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
7block_14_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �   �  *)
_class
loc:@block_14_expand/kernel*
dtype0*
_output_shapes
:
�
5block_14_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *�啽*)
_class
loc:@block_14_expand/kernel*
dtype0*
_output_shapes
: 
�
5block_14_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=*)
_class
loc:@block_14_expand/kernel*
dtype0*
_output_shapes
: 
�
?block_14_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_14_expand/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_14_expand/kernel*
dtype0*(
_output_shapes
:��
�
5block_14_expand/kernel/Initializer/random_uniform/subSub5block_14_expand/kernel/Initializer/random_uniform/max5block_14_expand/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_14_expand/kernel*
_output_shapes
: 
�
5block_14_expand/kernel/Initializer/random_uniform/mulMul?block_14_expand/kernel/Initializer/random_uniform/RandomUniform5block_14_expand/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_14_expand/kernel*(
_output_shapes
:��
�
1block_14_expand/kernel/Initializer/random_uniformAdd5block_14_expand/kernel/Initializer/random_uniform/mul5block_14_expand/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_14_expand/kernel*(
_output_shapes
:��
�
block_14_expand/kernelVarHandleOp*
shape:��*'
shared_nameblock_14_expand/kernel*)
_class
loc:@block_14_expand/kernel*
dtype0*
_output_shapes
: 
}
7block_14_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_14_expand/kernel*
_output_shapes
: 
�
block_14_expand/kernel/AssignAssignVariableOpblock_14_expand/kernel1block_14_expand/kernel/Initializer/random_uniform*)
_class
loc:@block_14_expand/kernel*
dtype0
�
*block_14_expand/kernel/Read/ReadVariableOpReadVariableOpblock_14_expand/kernel*)
_class
loc:@block_14_expand/kernel*
dtype0*(
_output_shapes
:��
n
block_14_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_14_expand/Conv2D/ReadVariableOpReadVariableOpblock_14_expand/kernel*
dtype0*(
_output_shapes
:��
�
block_14_expand/Conv2DConv2D"block_13_project_BN/FusedBatchNorm%block_14_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
)block_14_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?*+
_class!
loc:@block_14_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_14_expand_BN/gammaVarHandleOp*
shape:�*)
shared_nameblock_14_expand_BN/gamma*+
_class!
loc:@block_14_expand_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_14_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_14_expand_BN/gamma*
_output_shapes
: 
�
block_14_expand_BN/gamma/AssignAssignVariableOpblock_14_expand_BN/gamma)block_14_expand_BN/gamma/Initializer/ones*+
_class!
loc:@block_14_expand_BN/gamma*
dtype0
�
,block_14_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_14_expand_BN/gamma*+
_class!
loc:@block_14_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
)block_14_expand_BN/beta/Initializer/zerosConst*
valueB�*    **
_class 
loc:@block_14_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_14_expand_BN/betaVarHandleOp*
shape:�*(
shared_nameblock_14_expand_BN/beta**
_class 
loc:@block_14_expand_BN/beta*
dtype0*
_output_shapes
: 

8block_14_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_14_expand_BN/beta*
_output_shapes
: 
�
block_14_expand_BN/beta/AssignAssignVariableOpblock_14_expand_BN/beta)block_14_expand_BN/beta/Initializer/zeros**
_class 
loc:@block_14_expand_BN/beta*
dtype0
�
+block_14_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_14_expand_BN/beta**
_class 
loc:@block_14_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
0block_14_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *1
_class'
%#loc:@block_14_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_14_expand_BN/moving_meanVarHandleOp*
shape:�*/
shared_name block_14_expand_BN/moving_mean*1
_class'
%#loc:@block_14_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_14_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_14_expand_BN/moving_mean*
_output_shapes
: 
�
%block_14_expand_BN/moving_mean/AssignAssignVariableOpblock_14_expand_BN/moving_mean0block_14_expand_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_14_expand_BN/moving_mean*
dtype0
�
2block_14_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_14_expand_BN/moving_mean*1
_class'
%#loc:@block_14_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
3block_14_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*5
_class+
)'loc:@block_14_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
"block_14_expand_BN/moving_varianceVarHandleOp*
shape:�*3
shared_name$"block_14_expand_BN/moving_variance*5
_class+
)'loc:@block_14_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_14_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_14_expand_BN/moving_variance*
_output_shapes
: 
�
)block_14_expand_BN/moving_variance/AssignAssignVariableOp"block_14_expand_BN/moving_variance3block_14_expand_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_14_expand_BN/moving_variance*
dtype0
�
6block_14_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_14_expand_BN/moving_variance*5
_class+
)'loc:@block_14_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
w
!block_14_expand_BN/ReadVariableOpReadVariableOpblock_14_expand_BN/gamma*
dtype0*
_output_shapes	
:�
x
#block_14_expand_BN/ReadVariableOp_1ReadVariableOpblock_14_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
0block_14_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_14_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_14_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_14_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_14_expand_BN/FusedBatchNormFusedBatchNormblock_14_expand/Conv2D!block_14_expand_BN/ReadVariableOp#block_14_expand_BN/ReadVariableOp_10block_14_expand_BN/FusedBatchNorm/ReadVariableOp2block_14_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
]
block_14_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_14_expand_relu/Relu6Relu6!block_14_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
Dblock_14_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      �     *6
_class,
*(loc:@block_14_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Bblock_14_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *�׼*6
_class,
*(loc:@block_14_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Bblock_14_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *��<*6
_class,
*(loc:@block_14_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Lblock_14_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformDblock_14_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@block_14_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Bblock_14_depthwise/depthwise_kernel/Initializer/random_uniform/subSubBblock_14_depthwise/depthwise_kernel/Initializer/random_uniform/maxBblock_14_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@block_14_depthwise/depthwise_kernel*
_output_shapes
: 
�
Bblock_14_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulLblock_14_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformBblock_14_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@block_14_depthwise/depthwise_kernel*'
_output_shapes
:�
�
>block_14_depthwise/depthwise_kernel/Initializer/random_uniformAddBblock_14_depthwise/depthwise_kernel/Initializer/random_uniform/mulBblock_14_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@block_14_depthwise/depthwise_kernel*'
_output_shapes
:�
�
#block_14_depthwise/depthwise_kernelVarHandleOp*
shape:�*4
shared_name%#block_14_depthwise/depthwise_kernel*6
_class,
*(loc:@block_14_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Dblock_14_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp#block_14_depthwise/depthwise_kernel*
_output_shapes
: 
�
*block_14_depthwise/depthwise_kernel/AssignAssignVariableOp#block_14_depthwise/depthwise_kernel>block_14_depthwise/depthwise_kernel/Initializer/random_uniform*6
_class,
*(loc:@block_14_depthwise/depthwise_kernel*
dtype0
�
7block_14_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp#block_14_depthwise/depthwise_kernel*6
_class,
*(loc:@block_14_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
+block_14_depthwise/depthwise/ReadVariableOpReadVariableOp#block_14_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
{
"block_14_depthwise/depthwise/ShapeConst*%
valueB"      �     *
dtype0*
_output_shapes
:
{
*block_14_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_14_depthwise/depthwiseDepthwiseConv2dNativeblock_14_expand_relu/Relu6+block_14_depthwise/depthwise/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
,block_14_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*.
_class$
" loc:@block_14_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_14_depthwise_BN/gammaVarHandleOp*
shape:�*,
shared_nameblock_14_depthwise_BN/gamma*.
_class$
" loc:@block_14_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
<block_14_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_14_depthwise_BN/gamma*
_output_shapes
: 
�
"block_14_depthwise_BN/gamma/AssignAssignVariableOpblock_14_depthwise_BN/gamma,block_14_depthwise_BN/gamma/Initializer/ones*.
_class$
" loc:@block_14_depthwise_BN/gamma*
dtype0
�
/block_14_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_14_depthwise_BN/gamma*.
_class$
" loc:@block_14_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
,block_14_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *-
_class#
!loc:@block_14_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_14_depthwise_BN/betaVarHandleOp*
shape:�*+
shared_nameblock_14_depthwise_BN/beta*-
_class#
!loc:@block_14_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
;block_14_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_14_depthwise_BN/beta*
_output_shapes
: 
�
!block_14_depthwise_BN/beta/AssignAssignVariableOpblock_14_depthwise_BN/beta,block_14_depthwise_BN/beta/Initializer/zeros*-
_class#
!loc:@block_14_depthwise_BN/beta*
dtype0
�
.block_14_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_14_depthwise_BN/beta*-
_class#
!loc:@block_14_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
3block_14_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *4
_class*
(&loc:@block_14_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
!block_14_depthwise_BN/moving_meanVarHandleOp*
shape:�*2
shared_name#!block_14_depthwise_BN/moving_mean*4
_class*
(&loc:@block_14_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Bblock_14_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_14_depthwise_BN/moving_mean*
_output_shapes
: 
�
(block_14_depthwise_BN/moving_mean/AssignAssignVariableOp!block_14_depthwise_BN/moving_mean3block_14_depthwise_BN/moving_mean/Initializer/zeros*4
_class*
(&loc:@block_14_depthwise_BN/moving_mean*
dtype0
�
5block_14_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp!block_14_depthwise_BN/moving_mean*4
_class*
(&loc:@block_14_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
6block_14_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*8
_class.
,*loc:@block_14_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
%block_14_depthwise_BN/moving_varianceVarHandleOp*
shape:�*6
shared_name'%block_14_depthwise_BN/moving_variance*8
_class.
,*loc:@block_14_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Fblock_14_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%block_14_depthwise_BN/moving_variance*
_output_shapes
: 
�
,block_14_depthwise_BN/moving_variance/AssignAssignVariableOp%block_14_depthwise_BN/moving_variance6block_14_depthwise_BN/moving_variance/Initializer/ones*8
_class.
,*loc:@block_14_depthwise_BN/moving_variance*
dtype0
�
9block_14_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp%block_14_depthwise_BN/moving_variance*8
_class.
,*loc:@block_14_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
}
$block_14_depthwise_BN/ReadVariableOpReadVariableOpblock_14_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
~
&block_14_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_14_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
3block_14_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp!block_14_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_14_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp%block_14_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_14_depthwise_BN/FusedBatchNormFusedBatchNormblock_14_depthwise/depthwise$block_14_depthwise_BN/ReadVariableOp&block_14_depthwise_BN/ReadVariableOp_13block_14_depthwise_BN/FusedBatchNorm/ReadVariableOp5block_14_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
`
block_14_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_14_depthwise_relu/Relu6Relu6$block_14_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
8block_14_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �  �   **
_class 
loc:@block_14_project/kernel*
dtype0*
_output_shapes
:
�
6block_14_project/kernel/Initializer/random_uniform/minConst*
valueB
 *�啽**
_class 
loc:@block_14_project/kernel*
dtype0*
_output_shapes
: 
�
6block_14_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=**
_class 
loc:@block_14_project/kernel*
dtype0*
_output_shapes
: 
�
@block_14_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform8block_14_project/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@block_14_project/kernel*
dtype0*(
_output_shapes
:��
�
6block_14_project/kernel/Initializer/random_uniform/subSub6block_14_project/kernel/Initializer/random_uniform/max6block_14_project/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@block_14_project/kernel*
_output_shapes
: 
�
6block_14_project/kernel/Initializer/random_uniform/mulMul@block_14_project/kernel/Initializer/random_uniform/RandomUniform6block_14_project/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@block_14_project/kernel*(
_output_shapes
:��
�
2block_14_project/kernel/Initializer/random_uniformAdd6block_14_project/kernel/Initializer/random_uniform/mul6block_14_project/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@block_14_project/kernel*(
_output_shapes
:��
�
block_14_project/kernelVarHandleOp*
shape:��*(
shared_nameblock_14_project/kernel**
_class 
loc:@block_14_project/kernel*
dtype0*
_output_shapes
: 

8block_14_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_14_project/kernel*
_output_shapes
: 
�
block_14_project/kernel/AssignAssignVariableOpblock_14_project/kernel2block_14_project/kernel/Initializer/random_uniform**
_class 
loc:@block_14_project/kernel*
dtype0
�
+block_14_project/kernel/Read/ReadVariableOpReadVariableOpblock_14_project/kernel**
_class 
loc:@block_14_project/kernel*
dtype0*(
_output_shapes
:��
o
block_14_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
&block_14_project/Conv2D/ReadVariableOpReadVariableOpblock_14_project/kernel*
dtype0*(
_output_shapes
:��
�
block_14_project/Conv2DConv2Dblock_14_depthwise_relu/Relu6&block_14_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
*block_14_project_BN/gamma/Initializer/onesConst*
valueB�*  �?*,
_class"
 loc:@block_14_project_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_14_project_BN/gammaVarHandleOp*
shape:�**
shared_nameblock_14_project_BN/gamma*,
_class"
 loc:@block_14_project_BN/gamma*
dtype0*
_output_shapes
: 
�
:block_14_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_14_project_BN/gamma*
_output_shapes
: 
�
 block_14_project_BN/gamma/AssignAssignVariableOpblock_14_project_BN/gamma*block_14_project_BN/gamma/Initializer/ones*,
_class"
 loc:@block_14_project_BN/gamma*
dtype0
�
-block_14_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_14_project_BN/gamma*,
_class"
 loc:@block_14_project_BN/gamma*
dtype0*
_output_shapes	
:�
�
*block_14_project_BN/beta/Initializer/zerosConst*
valueB�*    *+
_class!
loc:@block_14_project_BN/beta*
dtype0*
_output_shapes	
:�
�
block_14_project_BN/betaVarHandleOp*
shape:�*)
shared_nameblock_14_project_BN/beta*+
_class!
loc:@block_14_project_BN/beta*
dtype0*
_output_shapes
: 
�
9block_14_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_14_project_BN/beta*
_output_shapes
: 
�
block_14_project_BN/beta/AssignAssignVariableOpblock_14_project_BN/beta*block_14_project_BN/beta/Initializer/zeros*+
_class!
loc:@block_14_project_BN/beta*
dtype0
�
,block_14_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_14_project_BN/beta*+
_class!
loc:@block_14_project_BN/beta*
dtype0*
_output_shapes	
:�
�
1block_14_project_BN/moving_mean/Initializer/zerosConst*
valueB�*    *2
_class(
&$loc:@block_14_project_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_14_project_BN/moving_meanVarHandleOp*
shape:�*0
shared_name!block_14_project_BN/moving_mean*2
_class(
&$loc:@block_14_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
@block_14_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_14_project_BN/moving_mean*
_output_shapes
: 
�
&block_14_project_BN/moving_mean/AssignAssignVariableOpblock_14_project_BN/moving_mean1block_14_project_BN/moving_mean/Initializer/zeros*2
_class(
&$loc:@block_14_project_BN/moving_mean*
dtype0
�
3block_14_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_14_project_BN/moving_mean*2
_class(
&$loc:@block_14_project_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
4block_14_project_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*6
_class,
*(loc:@block_14_project_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
#block_14_project_BN/moving_varianceVarHandleOp*
shape:�*4
shared_name%#block_14_project_BN/moving_variance*6
_class,
*(loc:@block_14_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Dblock_14_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#block_14_project_BN/moving_variance*
_output_shapes
: 
�
*block_14_project_BN/moving_variance/AssignAssignVariableOp#block_14_project_BN/moving_variance4block_14_project_BN/moving_variance/Initializer/ones*6
_class,
*(loc:@block_14_project_BN/moving_variance*
dtype0
�
7block_14_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp#block_14_project_BN/moving_variance*6
_class,
*(loc:@block_14_project_BN/moving_variance*
dtype0*
_output_shapes	
:�
y
"block_14_project_BN/ReadVariableOpReadVariableOpblock_14_project_BN/gamma*
dtype0*
_output_shapes	
:�
z
$block_14_project_BN/ReadVariableOp_1ReadVariableOpblock_14_project_BN/beta*
dtype0*
_output_shapes	
:�
�
1block_14_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_14_project_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
3block_14_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp#block_14_project_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
"block_14_project_BN/FusedBatchNormFusedBatchNormblock_14_project/Conv2D"block_14_project_BN/ReadVariableOp$block_14_project_BN/ReadVariableOp_11block_14_project_BN/FusedBatchNorm/ReadVariableOp3block_14_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
^
block_14_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_14_add/addAdd"block_13_project_BN/FusedBatchNorm"block_14_project_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
7block_15_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �   �  *)
_class
loc:@block_15_expand/kernel*
dtype0*
_output_shapes
:
�
5block_15_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *�啽*)
_class
loc:@block_15_expand/kernel*
dtype0*
_output_shapes
: 
�
5block_15_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=*)
_class
loc:@block_15_expand/kernel*
dtype0*
_output_shapes
: 
�
?block_15_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_15_expand/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_15_expand/kernel*
dtype0*(
_output_shapes
:��
�
5block_15_expand/kernel/Initializer/random_uniform/subSub5block_15_expand/kernel/Initializer/random_uniform/max5block_15_expand/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_15_expand/kernel*
_output_shapes
: 
�
5block_15_expand/kernel/Initializer/random_uniform/mulMul?block_15_expand/kernel/Initializer/random_uniform/RandomUniform5block_15_expand/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_15_expand/kernel*(
_output_shapes
:��
�
1block_15_expand/kernel/Initializer/random_uniformAdd5block_15_expand/kernel/Initializer/random_uniform/mul5block_15_expand/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_15_expand/kernel*(
_output_shapes
:��
�
block_15_expand/kernelVarHandleOp*
shape:��*'
shared_nameblock_15_expand/kernel*)
_class
loc:@block_15_expand/kernel*
dtype0*
_output_shapes
: 
}
7block_15_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_15_expand/kernel*
_output_shapes
: 
�
block_15_expand/kernel/AssignAssignVariableOpblock_15_expand/kernel1block_15_expand/kernel/Initializer/random_uniform*)
_class
loc:@block_15_expand/kernel*
dtype0
�
*block_15_expand/kernel/Read/ReadVariableOpReadVariableOpblock_15_expand/kernel*)
_class
loc:@block_15_expand/kernel*
dtype0*(
_output_shapes
:��
n
block_15_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_15_expand/Conv2D/ReadVariableOpReadVariableOpblock_15_expand/kernel*
dtype0*(
_output_shapes
:��
�
block_15_expand/Conv2DConv2Dblock_14_add/add%block_15_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
)block_15_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?*+
_class!
loc:@block_15_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_15_expand_BN/gammaVarHandleOp*
shape:�*)
shared_nameblock_15_expand_BN/gamma*+
_class!
loc:@block_15_expand_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_15_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_15_expand_BN/gamma*
_output_shapes
: 
�
block_15_expand_BN/gamma/AssignAssignVariableOpblock_15_expand_BN/gamma)block_15_expand_BN/gamma/Initializer/ones*+
_class!
loc:@block_15_expand_BN/gamma*
dtype0
�
,block_15_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_15_expand_BN/gamma*+
_class!
loc:@block_15_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
)block_15_expand_BN/beta/Initializer/zerosConst*
valueB�*    **
_class 
loc:@block_15_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_15_expand_BN/betaVarHandleOp*
shape:�*(
shared_nameblock_15_expand_BN/beta**
_class 
loc:@block_15_expand_BN/beta*
dtype0*
_output_shapes
: 

8block_15_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_15_expand_BN/beta*
_output_shapes
: 
�
block_15_expand_BN/beta/AssignAssignVariableOpblock_15_expand_BN/beta)block_15_expand_BN/beta/Initializer/zeros**
_class 
loc:@block_15_expand_BN/beta*
dtype0
�
+block_15_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_15_expand_BN/beta**
_class 
loc:@block_15_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
0block_15_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *1
_class'
%#loc:@block_15_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_15_expand_BN/moving_meanVarHandleOp*
shape:�*/
shared_name block_15_expand_BN/moving_mean*1
_class'
%#loc:@block_15_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_15_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_15_expand_BN/moving_mean*
_output_shapes
: 
�
%block_15_expand_BN/moving_mean/AssignAssignVariableOpblock_15_expand_BN/moving_mean0block_15_expand_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_15_expand_BN/moving_mean*
dtype0
�
2block_15_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_15_expand_BN/moving_mean*1
_class'
%#loc:@block_15_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
3block_15_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*5
_class+
)'loc:@block_15_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
"block_15_expand_BN/moving_varianceVarHandleOp*
shape:�*3
shared_name$"block_15_expand_BN/moving_variance*5
_class+
)'loc:@block_15_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_15_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_15_expand_BN/moving_variance*
_output_shapes
: 
�
)block_15_expand_BN/moving_variance/AssignAssignVariableOp"block_15_expand_BN/moving_variance3block_15_expand_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_15_expand_BN/moving_variance*
dtype0
�
6block_15_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_15_expand_BN/moving_variance*5
_class+
)'loc:@block_15_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
w
!block_15_expand_BN/ReadVariableOpReadVariableOpblock_15_expand_BN/gamma*
dtype0*
_output_shapes	
:�
x
#block_15_expand_BN/ReadVariableOp_1ReadVariableOpblock_15_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
0block_15_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_15_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_15_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_15_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_15_expand_BN/FusedBatchNormFusedBatchNormblock_15_expand/Conv2D!block_15_expand_BN/ReadVariableOp#block_15_expand_BN/ReadVariableOp_10block_15_expand_BN/FusedBatchNorm/ReadVariableOp2block_15_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
]
block_15_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_15_expand_relu/Relu6Relu6!block_15_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
Dblock_15_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      �     *6
_class,
*(loc:@block_15_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Bblock_15_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *�׼*6
_class,
*(loc:@block_15_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Bblock_15_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *��<*6
_class,
*(loc:@block_15_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Lblock_15_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformDblock_15_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@block_15_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Bblock_15_depthwise/depthwise_kernel/Initializer/random_uniform/subSubBblock_15_depthwise/depthwise_kernel/Initializer/random_uniform/maxBblock_15_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@block_15_depthwise/depthwise_kernel*
_output_shapes
: 
�
Bblock_15_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulLblock_15_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformBblock_15_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@block_15_depthwise/depthwise_kernel*'
_output_shapes
:�
�
>block_15_depthwise/depthwise_kernel/Initializer/random_uniformAddBblock_15_depthwise/depthwise_kernel/Initializer/random_uniform/mulBblock_15_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@block_15_depthwise/depthwise_kernel*'
_output_shapes
:�
�
#block_15_depthwise/depthwise_kernelVarHandleOp*
shape:�*4
shared_name%#block_15_depthwise/depthwise_kernel*6
_class,
*(loc:@block_15_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Dblock_15_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp#block_15_depthwise/depthwise_kernel*
_output_shapes
: 
�
*block_15_depthwise/depthwise_kernel/AssignAssignVariableOp#block_15_depthwise/depthwise_kernel>block_15_depthwise/depthwise_kernel/Initializer/random_uniform*6
_class,
*(loc:@block_15_depthwise/depthwise_kernel*
dtype0
�
7block_15_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp#block_15_depthwise/depthwise_kernel*6
_class,
*(loc:@block_15_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
+block_15_depthwise/depthwise/ReadVariableOpReadVariableOp#block_15_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
{
"block_15_depthwise/depthwise/ShapeConst*%
valueB"      �     *
dtype0*
_output_shapes
:
{
*block_15_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_15_depthwise/depthwiseDepthwiseConv2dNativeblock_15_expand_relu/Relu6+block_15_depthwise/depthwise/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
,block_15_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*.
_class$
" loc:@block_15_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_15_depthwise_BN/gammaVarHandleOp*
shape:�*,
shared_nameblock_15_depthwise_BN/gamma*.
_class$
" loc:@block_15_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
<block_15_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_15_depthwise_BN/gamma*
_output_shapes
: 
�
"block_15_depthwise_BN/gamma/AssignAssignVariableOpblock_15_depthwise_BN/gamma,block_15_depthwise_BN/gamma/Initializer/ones*.
_class$
" loc:@block_15_depthwise_BN/gamma*
dtype0
�
/block_15_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_15_depthwise_BN/gamma*.
_class$
" loc:@block_15_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
,block_15_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *-
_class#
!loc:@block_15_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_15_depthwise_BN/betaVarHandleOp*
shape:�*+
shared_nameblock_15_depthwise_BN/beta*-
_class#
!loc:@block_15_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
;block_15_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_15_depthwise_BN/beta*
_output_shapes
: 
�
!block_15_depthwise_BN/beta/AssignAssignVariableOpblock_15_depthwise_BN/beta,block_15_depthwise_BN/beta/Initializer/zeros*-
_class#
!loc:@block_15_depthwise_BN/beta*
dtype0
�
.block_15_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_15_depthwise_BN/beta*-
_class#
!loc:@block_15_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
3block_15_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *4
_class*
(&loc:@block_15_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
!block_15_depthwise_BN/moving_meanVarHandleOp*
shape:�*2
shared_name#!block_15_depthwise_BN/moving_mean*4
_class*
(&loc:@block_15_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Bblock_15_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_15_depthwise_BN/moving_mean*
_output_shapes
: 
�
(block_15_depthwise_BN/moving_mean/AssignAssignVariableOp!block_15_depthwise_BN/moving_mean3block_15_depthwise_BN/moving_mean/Initializer/zeros*4
_class*
(&loc:@block_15_depthwise_BN/moving_mean*
dtype0
�
5block_15_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp!block_15_depthwise_BN/moving_mean*4
_class*
(&loc:@block_15_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
6block_15_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*8
_class.
,*loc:@block_15_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
%block_15_depthwise_BN/moving_varianceVarHandleOp*
shape:�*6
shared_name'%block_15_depthwise_BN/moving_variance*8
_class.
,*loc:@block_15_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Fblock_15_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%block_15_depthwise_BN/moving_variance*
_output_shapes
: 
�
,block_15_depthwise_BN/moving_variance/AssignAssignVariableOp%block_15_depthwise_BN/moving_variance6block_15_depthwise_BN/moving_variance/Initializer/ones*8
_class.
,*loc:@block_15_depthwise_BN/moving_variance*
dtype0
�
9block_15_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp%block_15_depthwise_BN/moving_variance*8
_class.
,*loc:@block_15_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
}
$block_15_depthwise_BN/ReadVariableOpReadVariableOpblock_15_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
~
&block_15_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_15_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
3block_15_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp!block_15_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_15_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp%block_15_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_15_depthwise_BN/FusedBatchNormFusedBatchNormblock_15_depthwise/depthwise$block_15_depthwise_BN/ReadVariableOp&block_15_depthwise_BN/ReadVariableOp_13block_15_depthwise_BN/FusedBatchNorm/ReadVariableOp5block_15_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
`
block_15_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_15_depthwise_relu/Relu6Relu6$block_15_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
8block_15_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �  �   **
_class 
loc:@block_15_project/kernel*
dtype0*
_output_shapes
:
�
6block_15_project/kernel/Initializer/random_uniform/minConst*
valueB
 *�啽**
_class 
loc:@block_15_project/kernel*
dtype0*
_output_shapes
: 
�
6block_15_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=**
_class 
loc:@block_15_project/kernel*
dtype0*
_output_shapes
: 
�
@block_15_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform8block_15_project/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@block_15_project/kernel*
dtype0*(
_output_shapes
:��
�
6block_15_project/kernel/Initializer/random_uniform/subSub6block_15_project/kernel/Initializer/random_uniform/max6block_15_project/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@block_15_project/kernel*
_output_shapes
: 
�
6block_15_project/kernel/Initializer/random_uniform/mulMul@block_15_project/kernel/Initializer/random_uniform/RandomUniform6block_15_project/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@block_15_project/kernel*(
_output_shapes
:��
�
2block_15_project/kernel/Initializer/random_uniformAdd6block_15_project/kernel/Initializer/random_uniform/mul6block_15_project/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@block_15_project/kernel*(
_output_shapes
:��
�
block_15_project/kernelVarHandleOp*
shape:��*(
shared_nameblock_15_project/kernel**
_class 
loc:@block_15_project/kernel*
dtype0*
_output_shapes
: 

8block_15_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_15_project/kernel*
_output_shapes
: 
�
block_15_project/kernel/AssignAssignVariableOpblock_15_project/kernel2block_15_project/kernel/Initializer/random_uniform**
_class 
loc:@block_15_project/kernel*
dtype0
�
+block_15_project/kernel/Read/ReadVariableOpReadVariableOpblock_15_project/kernel**
_class 
loc:@block_15_project/kernel*
dtype0*(
_output_shapes
:��
o
block_15_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
&block_15_project/Conv2D/ReadVariableOpReadVariableOpblock_15_project/kernel*
dtype0*(
_output_shapes
:��
�
block_15_project/Conv2DConv2Dblock_15_depthwise_relu/Relu6&block_15_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
*block_15_project_BN/gamma/Initializer/onesConst*
valueB�*  �?*,
_class"
 loc:@block_15_project_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_15_project_BN/gammaVarHandleOp*
shape:�**
shared_nameblock_15_project_BN/gamma*,
_class"
 loc:@block_15_project_BN/gamma*
dtype0*
_output_shapes
: 
�
:block_15_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_15_project_BN/gamma*
_output_shapes
: 
�
 block_15_project_BN/gamma/AssignAssignVariableOpblock_15_project_BN/gamma*block_15_project_BN/gamma/Initializer/ones*,
_class"
 loc:@block_15_project_BN/gamma*
dtype0
�
-block_15_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_15_project_BN/gamma*,
_class"
 loc:@block_15_project_BN/gamma*
dtype0*
_output_shapes	
:�
�
*block_15_project_BN/beta/Initializer/zerosConst*
valueB�*    *+
_class!
loc:@block_15_project_BN/beta*
dtype0*
_output_shapes	
:�
�
block_15_project_BN/betaVarHandleOp*
shape:�*)
shared_nameblock_15_project_BN/beta*+
_class!
loc:@block_15_project_BN/beta*
dtype0*
_output_shapes
: 
�
9block_15_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_15_project_BN/beta*
_output_shapes
: 
�
block_15_project_BN/beta/AssignAssignVariableOpblock_15_project_BN/beta*block_15_project_BN/beta/Initializer/zeros*+
_class!
loc:@block_15_project_BN/beta*
dtype0
�
,block_15_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_15_project_BN/beta*+
_class!
loc:@block_15_project_BN/beta*
dtype0*
_output_shapes	
:�
�
1block_15_project_BN/moving_mean/Initializer/zerosConst*
valueB�*    *2
_class(
&$loc:@block_15_project_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_15_project_BN/moving_meanVarHandleOp*
shape:�*0
shared_name!block_15_project_BN/moving_mean*2
_class(
&$loc:@block_15_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
@block_15_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_15_project_BN/moving_mean*
_output_shapes
: 
�
&block_15_project_BN/moving_mean/AssignAssignVariableOpblock_15_project_BN/moving_mean1block_15_project_BN/moving_mean/Initializer/zeros*2
_class(
&$loc:@block_15_project_BN/moving_mean*
dtype0
�
3block_15_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_15_project_BN/moving_mean*2
_class(
&$loc:@block_15_project_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
4block_15_project_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*6
_class,
*(loc:@block_15_project_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
#block_15_project_BN/moving_varianceVarHandleOp*
shape:�*4
shared_name%#block_15_project_BN/moving_variance*6
_class,
*(loc:@block_15_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Dblock_15_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#block_15_project_BN/moving_variance*
_output_shapes
: 
�
*block_15_project_BN/moving_variance/AssignAssignVariableOp#block_15_project_BN/moving_variance4block_15_project_BN/moving_variance/Initializer/ones*6
_class,
*(loc:@block_15_project_BN/moving_variance*
dtype0
�
7block_15_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp#block_15_project_BN/moving_variance*6
_class,
*(loc:@block_15_project_BN/moving_variance*
dtype0*
_output_shapes	
:�
y
"block_15_project_BN/ReadVariableOpReadVariableOpblock_15_project_BN/gamma*
dtype0*
_output_shapes	
:�
z
$block_15_project_BN/ReadVariableOp_1ReadVariableOpblock_15_project_BN/beta*
dtype0*
_output_shapes	
:�
�
1block_15_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_15_project_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
3block_15_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp#block_15_project_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
"block_15_project_BN/FusedBatchNormFusedBatchNormblock_15_project/Conv2D"block_15_project_BN/ReadVariableOp$block_15_project_BN/ReadVariableOp_11block_15_project_BN/FusedBatchNorm/ReadVariableOp3block_15_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
^
block_15_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_15_add/addAddblock_14_add/add"block_15_project_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
7block_16_expand/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �   �  *)
_class
loc:@block_16_expand/kernel*
dtype0*
_output_shapes
:
�
5block_16_expand/kernel/Initializer/random_uniform/minConst*
valueB
 *�啽*)
_class
loc:@block_16_expand/kernel*
dtype0*
_output_shapes
: 
�
5block_16_expand/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=*)
_class
loc:@block_16_expand/kernel*
dtype0*
_output_shapes
: 
�
?block_16_expand/kernel/Initializer/random_uniform/RandomUniformRandomUniform7block_16_expand/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@block_16_expand/kernel*
dtype0*(
_output_shapes
:��
�
5block_16_expand/kernel/Initializer/random_uniform/subSub5block_16_expand/kernel/Initializer/random_uniform/max5block_16_expand/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_16_expand/kernel*
_output_shapes
: 
�
5block_16_expand/kernel/Initializer/random_uniform/mulMul?block_16_expand/kernel/Initializer/random_uniform/RandomUniform5block_16_expand/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@block_16_expand/kernel*(
_output_shapes
:��
�
1block_16_expand/kernel/Initializer/random_uniformAdd5block_16_expand/kernel/Initializer/random_uniform/mul5block_16_expand/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@block_16_expand/kernel*(
_output_shapes
:��
�
block_16_expand/kernelVarHandleOp*
shape:��*'
shared_nameblock_16_expand/kernel*)
_class
loc:@block_16_expand/kernel*
dtype0*
_output_shapes
: 
}
7block_16_expand/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_16_expand/kernel*
_output_shapes
: 
�
block_16_expand/kernel/AssignAssignVariableOpblock_16_expand/kernel1block_16_expand/kernel/Initializer/random_uniform*)
_class
loc:@block_16_expand/kernel*
dtype0
�
*block_16_expand/kernel/Read/ReadVariableOpReadVariableOpblock_16_expand/kernel*)
_class
loc:@block_16_expand/kernel*
dtype0*(
_output_shapes
:��
n
block_16_expand/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%block_16_expand/Conv2D/ReadVariableOpReadVariableOpblock_16_expand/kernel*
dtype0*(
_output_shapes
:��
�
block_16_expand/Conv2DConv2Dblock_15_add/add%block_16_expand/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
)block_16_expand_BN/gamma/Initializer/onesConst*
valueB�*  �?*+
_class!
loc:@block_16_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_16_expand_BN/gammaVarHandleOp*
shape:�*)
shared_nameblock_16_expand_BN/gamma*+
_class!
loc:@block_16_expand_BN/gamma*
dtype0*
_output_shapes
: 
�
9block_16_expand_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_16_expand_BN/gamma*
_output_shapes
: 
�
block_16_expand_BN/gamma/AssignAssignVariableOpblock_16_expand_BN/gamma)block_16_expand_BN/gamma/Initializer/ones*+
_class!
loc:@block_16_expand_BN/gamma*
dtype0
�
,block_16_expand_BN/gamma/Read/ReadVariableOpReadVariableOpblock_16_expand_BN/gamma*+
_class!
loc:@block_16_expand_BN/gamma*
dtype0*
_output_shapes	
:�
�
)block_16_expand_BN/beta/Initializer/zerosConst*
valueB�*    **
_class 
loc:@block_16_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
block_16_expand_BN/betaVarHandleOp*
shape:�*(
shared_nameblock_16_expand_BN/beta**
_class 
loc:@block_16_expand_BN/beta*
dtype0*
_output_shapes
: 

8block_16_expand_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_16_expand_BN/beta*
_output_shapes
: 
�
block_16_expand_BN/beta/AssignAssignVariableOpblock_16_expand_BN/beta)block_16_expand_BN/beta/Initializer/zeros**
_class 
loc:@block_16_expand_BN/beta*
dtype0
�
+block_16_expand_BN/beta/Read/ReadVariableOpReadVariableOpblock_16_expand_BN/beta**
_class 
loc:@block_16_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
0block_16_expand_BN/moving_mean/Initializer/zerosConst*
valueB�*    *1
_class'
%#loc:@block_16_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_16_expand_BN/moving_meanVarHandleOp*
shape:�*/
shared_name block_16_expand_BN/moving_mean*1
_class'
%#loc:@block_16_expand_BN/moving_mean*
dtype0*
_output_shapes
: 
�
?block_16_expand_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_16_expand_BN/moving_mean*
_output_shapes
: 
�
%block_16_expand_BN/moving_mean/AssignAssignVariableOpblock_16_expand_BN/moving_mean0block_16_expand_BN/moving_mean/Initializer/zeros*1
_class'
%#loc:@block_16_expand_BN/moving_mean*
dtype0
�
2block_16_expand_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_16_expand_BN/moving_mean*1
_class'
%#loc:@block_16_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
3block_16_expand_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*5
_class+
)'loc:@block_16_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
"block_16_expand_BN/moving_varianceVarHandleOp*
shape:�*3
shared_name$"block_16_expand_BN/moving_variance*5
_class+
)'loc:@block_16_expand_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Cblock_16_expand_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp"block_16_expand_BN/moving_variance*
_output_shapes
: 
�
)block_16_expand_BN/moving_variance/AssignAssignVariableOp"block_16_expand_BN/moving_variance3block_16_expand_BN/moving_variance/Initializer/ones*5
_class+
)'loc:@block_16_expand_BN/moving_variance*
dtype0
�
6block_16_expand_BN/moving_variance/Read/ReadVariableOpReadVariableOp"block_16_expand_BN/moving_variance*5
_class+
)'loc:@block_16_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
w
!block_16_expand_BN/ReadVariableOpReadVariableOpblock_16_expand_BN/gamma*
dtype0*
_output_shapes	
:�
x
#block_16_expand_BN/ReadVariableOp_1ReadVariableOpblock_16_expand_BN/beta*
dtype0*
_output_shapes	
:�
�
0block_16_expand_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_16_expand_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
2block_16_expand_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp"block_16_expand_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
!block_16_expand_BN/FusedBatchNormFusedBatchNormblock_16_expand/Conv2D!block_16_expand_BN/ReadVariableOp#block_16_expand_BN/ReadVariableOp_10block_16_expand_BN/FusedBatchNorm/ReadVariableOp2block_16_expand_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
]
block_16_expand_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_16_expand_relu/Relu6Relu6!block_16_expand_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
Dblock_16_depthwise/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"      �     *6
_class,
*(loc:@block_16_depthwise/depthwise_kernel*
dtype0*
_output_shapes
:
�
Bblock_16_depthwise/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *�׼*6
_class,
*(loc:@block_16_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Bblock_16_depthwise/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *��<*6
_class,
*(loc:@block_16_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Lblock_16_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniformDblock_16_depthwise/depthwise_kernel/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@block_16_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
Bblock_16_depthwise/depthwise_kernel/Initializer/random_uniform/subSubBblock_16_depthwise/depthwise_kernel/Initializer/random_uniform/maxBblock_16_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@block_16_depthwise/depthwise_kernel*
_output_shapes
: 
�
Bblock_16_depthwise/depthwise_kernel/Initializer/random_uniform/mulMulLblock_16_depthwise/depthwise_kernel/Initializer/random_uniform/RandomUniformBblock_16_depthwise/depthwise_kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@block_16_depthwise/depthwise_kernel*'
_output_shapes
:�
�
>block_16_depthwise/depthwise_kernel/Initializer/random_uniformAddBblock_16_depthwise/depthwise_kernel/Initializer/random_uniform/mulBblock_16_depthwise/depthwise_kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@block_16_depthwise/depthwise_kernel*'
_output_shapes
:�
�
#block_16_depthwise/depthwise_kernelVarHandleOp*
shape:�*4
shared_name%#block_16_depthwise/depthwise_kernel*6
_class,
*(loc:@block_16_depthwise/depthwise_kernel*
dtype0*
_output_shapes
: 
�
Dblock_16_depthwise/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp#block_16_depthwise/depthwise_kernel*
_output_shapes
: 
�
*block_16_depthwise/depthwise_kernel/AssignAssignVariableOp#block_16_depthwise/depthwise_kernel>block_16_depthwise/depthwise_kernel/Initializer/random_uniform*6
_class,
*(loc:@block_16_depthwise/depthwise_kernel*
dtype0
�
7block_16_depthwise/depthwise_kernel/Read/ReadVariableOpReadVariableOp#block_16_depthwise/depthwise_kernel*6
_class,
*(loc:@block_16_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
�
+block_16_depthwise/depthwise/ReadVariableOpReadVariableOp#block_16_depthwise/depthwise_kernel*
dtype0*'
_output_shapes
:�
{
"block_16_depthwise/depthwise/ShapeConst*%
valueB"      �     *
dtype0*
_output_shapes
:
{
*block_16_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
block_16_depthwise/depthwiseDepthwiseConv2dNativeblock_16_expand_relu/Relu6+block_16_depthwise/depthwise/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
,block_16_depthwise_BN/gamma/Initializer/onesConst*
valueB�*  �?*.
_class$
" loc:@block_16_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_16_depthwise_BN/gammaVarHandleOp*
shape:�*,
shared_nameblock_16_depthwise_BN/gamma*.
_class$
" loc:@block_16_depthwise_BN/gamma*
dtype0*
_output_shapes
: 
�
<block_16_depthwise_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_16_depthwise_BN/gamma*
_output_shapes
: 
�
"block_16_depthwise_BN/gamma/AssignAssignVariableOpblock_16_depthwise_BN/gamma,block_16_depthwise_BN/gamma/Initializer/ones*.
_class$
" loc:@block_16_depthwise_BN/gamma*
dtype0
�
/block_16_depthwise_BN/gamma/Read/ReadVariableOpReadVariableOpblock_16_depthwise_BN/gamma*.
_class$
" loc:@block_16_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
�
,block_16_depthwise_BN/beta/Initializer/zerosConst*
valueB�*    *-
_class#
!loc:@block_16_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
block_16_depthwise_BN/betaVarHandleOp*
shape:�*+
shared_nameblock_16_depthwise_BN/beta*-
_class#
!loc:@block_16_depthwise_BN/beta*
dtype0*
_output_shapes
: 
�
;block_16_depthwise_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_16_depthwise_BN/beta*
_output_shapes
: 
�
!block_16_depthwise_BN/beta/AssignAssignVariableOpblock_16_depthwise_BN/beta,block_16_depthwise_BN/beta/Initializer/zeros*-
_class#
!loc:@block_16_depthwise_BN/beta*
dtype0
�
.block_16_depthwise_BN/beta/Read/ReadVariableOpReadVariableOpblock_16_depthwise_BN/beta*-
_class#
!loc:@block_16_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
3block_16_depthwise_BN/moving_mean/Initializer/zerosConst*
valueB�*    *4
_class*
(&loc:@block_16_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
!block_16_depthwise_BN/moving_meanVarHandleOp*
shape:�*2
shared_name#!block_16_depthwise_BN/moving_mean*4
_class*
(&loc:@block_16_depthwise_BN/moving_mean*
dtype0*
_output_shapes
: 
�
Bblock_16_depthwise_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!block_16_depthwise_BN/moving_mean*
_output_shapes
: 
�
(block_16_depthwise_BN/moving_mean/AssignAssignVariableOp!block_16_depthwise_BN/moving_mean3block_16_depthwise_BN/moving_mean/Initializer/zeros*4
_class*
(&loc:@block_16_depthwise_BN/moving_mean*
dtype0
�
5block_16_depthwise_BN/moving_mean/Read/ReadVariableOpReadVariableOp!block_16_depthwise_BN/moving_mean*4
_class*
(&loc:@block_16_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
6block_16_depthwise_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*8
_class.
,*loc:@block_16_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
%block_16_depthwise_BN/moving_varianceVarHandleOp*
shape:�*6
shared_name'%block_16_depthwise_BN/moving_variance*8
_class.
,*loc:@block_16_depthwise_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Fblock_16_depthwise_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%block_16_depthwise_BN/moving_variance*
_output_shapes
: 
�
,block_16_depthwise_BN/moving_variance/AssignAssignVariableOp%block_16_depthwise_BN/moving_variance6block_16_depthwise_BN/moving_variance/Initializer/ones*8
_class.
,*loc:@block_16_depthwise_BN/moving_variance*
dtype0
�
9block_16_depthwise_BN/moving_variance/Read/ReadVariableOpReadVariableOp%block_16_depthwise_BN/moving_variance*8
_class.
,*loc:@block_16_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
}
$block_16_depthwise_BN/ReadVariableOpReadVariableOpblock_16_depthwise_BN/gamma*
dtype0*
_output_shapes	
:�
~
&block_16_depthwise_BN/ReadVariableOp_1ReadVariableOpblock_16_depthwise_BN/beta*
dtype0*
_output_shapes	
:�
�
3block_16_depthwise_BN/FusedBatchNorm/ReadVariableOpReadVariableOp!block_16_depthwise_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
5block_16_depthwise_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp%block_16_depthwise_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
$block_16_depthwise_BN/FusedBatchNormFusedBatchNormblock_16_depthwise/depthwise$block_16_depthwise_BN/ReadVariableOp&block_16_depthwise_BN/ReadVariableOp_13block_16_depthwise_BN/FusedBatchNorm/ReadVariableOp5block_16_depthwise_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
`
block_16_depthwise_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
block_16_depthwise_relu/Relu6Relu6$block_16_depthwise_BN/FusedBatchNorm*
T0*0
_output_shapes
:����������
�
8block_16_project/kernel/Initializer/random_uniform/shapeConst*%
valueB"      �  @  **
_class 
loc:@block_16_project/kernel*
dtype0*
_output_shapes
:
�
6block_16_project/kernel/Initializer/random_uniform/minConst*
valueB
 *�7��**
_class 
loc:@block_16_project/kernel*
dtype0*
_output_shapes
: 
�
6block_16_project/kernel/Initializer/random_uniform/maxConst*
valueB
 *�7�=**
_class 
loc:@block_16_project/kernel*
dtype0*
_output_shapes
: 
�
@block_16_project/kernel/Initializer/random_uniform/RandomUniformRandomUniform8block_16_project/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@block_16_project/kernel*
dtype0*(
_output_shapes
:��
�
6block_16_project/kernel/Initializer/random_uniform/subSub6block_16_project/kernel/Initializer/random_uniform/max6block_16_project/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@block_16_project/kernel*
_output_shapes
: 
�
6block_16_project/kernel/Initializer/random_uniform/mulMul@block_16_project/kernel/Initializer/random_uniform/RandomUniform6block_16_project/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@block_16_project/kernel*(
_output_shapes
:��
�
2block_16_project/kernel/Initializer/random_uniformAdd6block_16_project/kernel/Initializer/random_uniform/mul6block_16_project/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@block_16_project/kernel*(
_output_shapes
:��
�
block_16_project/kernelVarHandleOp*
shape:��*(
shared_nameblock_16_project/kernel**
_class 
loc:@block_16_project/kernel*
dtype0*
_output_shapes
: 

8block_16_project/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_16_project/kernel*
_output_shapes
: 
�
block_16_project/kernel/AssignAssignVariableOpblock_16_project/kernel2block_16_project/kernel/Initializer/random_uniform**
_class 
loc:@block_16_project/kernel*
dtype0
�
+block_16_project/kernel/Read/ReadVariableOpReadVariableOpblock_16_project/kernel**
_class 
loc:@block_16_project/kernel*
dtype0*(
_output_shapes
:��
o
block_16_project/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
&block_16_project/Conv2D/ReadVariableOpReadVariableOpblock_16_project/kernel*
dtype0*(
_output_shapes
:��
�
block_16_project/Conv2DConv2Dblock_16_depthwise_relu/Relu6&block_16_project/Conv2D/ReadVariableOp*
paddingSAME*
T0*
strides
*0
_output_shapes
:����������
�
*block_16_project_BN/gamma/Initializer/onesConst*
valueB�*  �?*,
_class"
 loc:@block_16_project_BN/gamma*
dtype0*
_output_shapes	
:�
�
block_16_project_BN/gammaVarHandleOp*
shape:�**
shared_nameblock_16_project_BN/gamma*,
_class"
 loc:@block_16_project_BN/gamma*
dtype0*
_output_shapes
: 
�
:block_16_project_BN/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_16_project_BN/gamma*
_output_shapes
: 
�
 block_16_project_BN/gamma/AssignAssignVariableOpblock_16_project_BN/gamma*block_16_project_BN/gamma/Initializer/ones*,
_class"
 loc:@block_16_project_BN/gamma*
dtype0
�
-block_16_project_BN/gamma/Read/ReadVariableOpReadVariableOpblock_16_project_BN/gamma*,
_class"
 loc:@block_16_project_BN/gamma*
dtype0*
_output_shapes	
:�
�
*block_16_project_BN/beta/Initializer/zerosConst*
valueB�*    *+
_class!
loc:@block_16_project_BN/beta*
dtype0*
_output_shapes	
:�
�
block_16_project_BN/betaVarHandleOp*
shape:�*)
shared_nameblock_16_project_BN/beta*+
_class!
loc:@block_16_project_BN/beta*
dtype0*
_output_shapes
: 
�
9block_16_project_BN/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_16_project_BN/beta*
_output_shapes
: 
�
block_16_project_BN/beta/AssignAssignVariableOpblock_16_project_BN/beta*block_16_project_BN/beta/Initializer/zeros*+
_class!
loc:@block_16_project_BN/beta*
dtype0
�
,block_16_project_BN/beta/Read/ReadVariableOpReadVariableOpblock_16_project_BN/beta*+
_class!
loc:@block_16_project_BN/beta*
dtype0*
_output_shapes	
:�
�
1block_16_project_BN/moving_mean/Initializer/zerosConst*
valueB�*    *2
_class(
&$loc:@block_16_project_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
block_16_project_BN/moving_meanVarHandleOp*
shape:�*0
shared_name!block_16_project_BN/moving_mean*2
_class(
&$loc:@block_16_project_BN/moving_mean*
dtype0*
_output_shapes
: 
�
@block_16_project_BN/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock_16_project_BN/moving_mean*
_output_shapes
: 
�
&block_16_project_BN/moving_mean/AssignAssignVariableOpblock_16_project_BN/moving_mean1block_16_project_BN/moving_mean/Initializer/zeros*2
_class(
&$loc:@block_16_project_BN/moving_mean*
dtype0
�
3block_16_project_BN/moving_mean/Read/ReadVariableOpReadVariableOpblock_16_project_BN/moving_mean*2
_class(
&$loc:@block_16_project_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
4block_16_project_BN/moving_variance/Initializer/onesConst*
valueB�*  �?*6
_class,
*(loc:@block_16_project_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
#block_16_project_BN/moving_varianceVarHandleOp*
shape:�*4
shared_name%#block_16_project_BN/moving_variance*6
_class,
*(loc:@block_16_project_BN/moving_variance*
dtype0*
_output_shapes
: 
�
Dblock_16_project_BN/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#block_16_project_BN/moving_variance*
_output_shapes
: 
�
*block_16_project_BN/moving_variance/AssignAssignVariableOp#block_16_project_BN/moving_variance4block_16_project_BN/moving_variance/Initializer/ones*6
_class,
*(loc:@block_16_project_BN/moving_variance*
dtype0
�
7block_16_project_BN/moving_variance/Read/ReadVariableOpReadVariableOp#block_16_project_BN/moving_variance*6
_class,
*(loc:@block_16_project_BN/moving_variance*
dtype0*
_output_shapes	
:�
y
"block_16_project_BN/ReadVariableOpReadVariableOpblock_16_project_BN/gamma*
dtype0*
_output_shapes	
:�
z
$block_16_project_BN/ReadVariableOp_1ReadVariableOpblock_16_project_BN/beta*
dtype0*
_output_shapes	
:�
�
1block_16_project_BN/FusedBatchNorm/ReadVariableOpReadVariableOpblock_16_project_BN/moving_mean*
dtype0*
_output_shapes	
:�
�
3block_16_project_BN/FusedBatchNorm/ReadVariableOp_1ReadVariableOp#block_16_project_BN/moving_variance*
dtype0*
_output_shapes	
:�
�
"block_16_project_BN/FusedBatchNormFusedBatchNormblock_16_project/Conv2D"block_16_project_BN/ReadVariableOp$block_16_project_BN/ReadVariableOp_11block_16_project_BN/FusedBatchNorm/ReadVariableOp3block_16_project_BN/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������:�:�:�:�*
is_training( 
^
block_16_project_BN/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
.Conv_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @     * 
_class
loc:@Conv_1/kernel*
dtype0*
_output_shapes
:
�
,Conv_1/kernel/Initializer/random_uniform/minConst*
valueB
 *��z�* 
_class
loc:@Conv_1/kernel*
dtype0*
_output_shapes
: 
�
,Conv_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *��z=* 
_class
loc:@Conv_1/kernel*
dtype0*
_output_shapes
: 
�
6Conv_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform.Conv_1/kernel/Initializer/random_uniform/shape*
T0* 
_class
loc:@Conv_1/kernel*
dtype0*(
_output_shapes
:��

�
,Conv_1/kernel/Initializer/random_uniform/subSub,Conv_1/kernel/Initializer/random_uniform/max,Conv_1/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@Conv_1/kernel*
_output_shapes
: 
�
,Conv_1/kernel/Initializer/random_uniform/mulMul6Conv_1/kernel/Initializer/random_uniform/RandomUniform,Conv_1/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@Conv_1/kernel*(
_output_shapes
:��

�
(Conv_1/kernel/Initializer/random_uniformAdd,Conv_1/kernel/Initializer/random_uniform/mul,Conv_1/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@Conv_1/kernel*(
_output_shapes
:��

�
Conv_1/kernelVarHandleOp*
shape:��
*
shared_nameConv_1/kernel* 
_class
loc:@Conv_1/kernel*
dtype0*
_output_shapes
: 
k
.Conv_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpConv_1/kernel*
_output_shapes
: 
�
Conv_1/kernel/AssignAssignVariableOpConv_1/kernel(Conv_1/kernel/Initializer/random_uniform* 
_class
loc:@Conv_1/kernel*
dtype0
�
!Conv_1/kernel/Read/ReadVariableOpReadVariableOpConv_1/kernel* 
_class
loc:@Conv_1/kernel*
dtype0*(
_output_shapes
:��

e
Conv_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
t
Conv_1/Conv2D/ReadVariableOpReadVariableOpConv_1/kernel*
dtype0*(
_output_shapes
:��

�
Conv_1/Conv2DConv2D"block_16_project_BN/FusedBatchNormConv_1/Conv2D/ReadVariableOp*
paddingVALID*
T0*
strides
*0
_output_shapes
:����������

�
0Conv_1_bn/gamma/Initializer/ones/shape_as_tensorConst*
valueB:�
*"
_class
loc:@Conv_1_bn/gamma*
dtype0*
_output_shapes
:
�
&Conv_1_bn/gamma/Initializer/ones/ConstConst*
valueB
 *  �?*"
_class
loc:@Conv_1_bn/gamma*
dtype0*
_output_shapes
: 
�
 Conv_1_bn/gamma/Initializer/onesFill0Conv_1_bn/gamma/Initializer/ones/shape_as_tensor&Conv_1_bn/gamma/Initializer/ones/Const*
T0*"
_class
loc:@Conv_1_bn/gamma*
_output_shapes	
:�

�
Conv_1_bn/gammaVarHandleOp*
shape:�
* 
shared_nameConv_1_bn/gamma*"
_class
loc:@Conv_1_bn/gamma*
dtype0*
_output_shapes
: 
o
0Conv_1_bn/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpConv_1_bn/gamma*
_output_shapes
: 
�
Conv_1_bn/gamma/AssignAssignVariableOpConv_1_bn/gamma Conv_1_bn/gamma/Initializer/ones*"
_class
loc:@Conv_1_bn/gamma*
dtype0
�
#Conv_1_bn/gamma/Read/ReadVariableOpReadVariableOpConv_1_bn/gamma*"
_class
loc:@Conv_1_bn/gamma*
dtype0*
_output_shapes	
:�

�
0Conv_1_bn/beta/Initializer/zeros/shape_as_tensorConst*
valueB:�
*!
_class
loc:@Conv_1_bn/beta*
dtype0*
_output_shapes
:
�
&Conv_1_bn/beta/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@Conv_1_bn/beta*
dtype0*
_output_shapes
: 
�
 Conv_1_bn/beta/Initializer/zerosFill0Conv_1_bn/beta/Initializer/zeros/shape_as_tensor&Conv_1_bn/beta/Initializer/zeros/Const*
T0*!
_class
loc:@Conv_1_bn/beta*
_output_shapes	
:�

�
Conv_1_bn/betaVarHandleOp*
shape:�
*
shared_nameConv_1_bn/beta*!
_class
loc:@Conv_1_bn/beta*
dtype0*
_output_shapes
: 
m
/Conv_1_bn/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpConv_1_bn/beta*
_output_shapes
: 
�
Conv_1_bn/beta/AssignAssignVariableOpConv_1_bn/beta Conv_1_bn/beta/Initializer/zeros*!
_class
loc:@Conv_1_bn/beta*
dtype0
�
"Conv_1_bn/beta/Read/ReadVariableOpReadVariableOpConv_1_bn/beta*!
_class
loc:@Conv_1_bn/beta*
dtype0*
_output_shapes	
:�

�
7Conv_1_bn/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:�
*(
_class
loc:@Conv_1_bn/moving_mean*
dtype0*
_output_shapes
:
�
-Conv_1_bn/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *(
_class
loc:@Conv_1_bn/moving_mean*
dtype0*
_output_shapes
: 
�
'Conv_1_bn/moving_mean/Initializer/zerosFill7Conv_1_bn/moving_mean/Initializer/zeros/shape_as_tensor-Conv_1_bn/moving_mean/Initializer/zeros/Const*
T0*(
_class
loc:@Conv_1_bn/moving_mean*
_output_shapes	
:�

�
Conv_1_bn/moving_meanVarHandleOp*
shape:�
*&
shared_nameConv_1_bn/moving_mean*(
_class
loc:@Conv_1_bn/moving_mean*
dtype0*
_output_shapes
: 
{
6Conv_1_bn/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpConv_1_bn/moving_mean*
_output_shapes
: 
�
Conv_1_bn/moving_mean/AssignAssignVariableOpConv_1_bn/moving_mean'Conv_1_bn/moving_mean/Initializer/zeros*(
_class
loc:@Conv_1_bn/moving_mean*
dtype0
�
)Conv_1_bn/moving_mean/Read/ReadVariableOpReadVariableOpConv_1_bn/moving_mean*(
_class
loc:@Conv_1_bn/moving_mean*
dtype0*
_output_shapes	
:�

�
:Conv_1_bn/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:�
*,
_class"
 loc:@Conv_1_bn/moving_variance*
dtype0*
_output_shapes
:
�
0Conv_1_bn/moving_variance/Initializer/ones/ConstConst*
valueB
 *  �?*,
_class"
 loc:@Conv_1_bn/moving_variance*
dtype0*
_output_shapes
: 
�
*Conv_1_bn/moving_variance/Initializer/onesFill:Conv_1_bn/moving_variance/Initializer/ones/shape_as_tensor0Conv_1_bn/moving_variance/Initializer/ones/Const*
T0*,
_class"
 loc:@Conv_1_bn/moving_variance*
_output_shapes	
:�

�
Conv_1_bn/moving_varianceVarHandleOp*
shape:�
**
shared_nameConv_1_bn/moving_variance*,
_class"
 loc:@Conv_1_bn/moving_variance*
dtype0*
_output_shapes
: 
�
:Conv_1_bn/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOpConv_1_bn/moving_variance*
_output_shapes
: 
�
 Conv_1_bn/moving_variance/AssignAssignVariableOpConv_1_bn/moving_variance*Conv_1_bn/moving_variance/Initializer/ones*,
_class"
 loc:@Conv_1_bn/moving_variance*
dtype0
�
-Conv_1_bn/moving_variance/Read/ReadVariableOpReadVariableOpConv_1_bn/moving_variance*,
_class"
 loc:@Conv_1_bn/moving_variance*
dtype0*
_output_shapes	
:�

e
Conv_1_bn/ReadVariableOpReadVariableOpConv_1_bn/gamma*
dtype0*
_output_shapes	
:�

f
Conv_1_bn/ReadVariableOp_1ReadVariableOpConv_1_bn/beta*
dtype0*
_output_shapes	
:�

z
'Conv_1_bn/FusedBatchNorm/ReadVariableOpReadVariableOpConv_1_bn/moving_mean*
dtype0*
_output_shapes	
:�

�
)Conv_1_bn/FusedBatchNorm/ReadVariableOp_1ReadVariableOpConv_1_bn/moving_variance*
dtype0*
_output_shapes	
:�

�
Conv_1_bn/FusedBatchNormFusedBatchNormConv_1/Conv2DConv_1_bn/ReadVariableOpConv_1_bn/ReadVariableOp_1'Conv_1_bn/FusedBatchNorm/ReadVariableOp)Conv_1_bn/FusedBatchNorm/ReadVariableOp_1*
epsilon%o�:*
T0*L
_output_shapes:
8:����������
:�
:�
:�
:�
*
is_training( 
T
Conv_1_bn/ConstConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
l
out_relu/Relu6Relu6Conv_1_bn/FusedBatchNorm*
T0*0
_output_shapes
:����������

�
/global_average_pooling2d/Mean/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
�
global_average_pooling2d/MeanMeanout_relu/Relu6/global_average_pooling2d/Mean/reduction_indices*
T0*(
_output_shapes
:����������

:
predict/group_depsNoOp^global_average_pooling2d/Mean
Z
ConstConst"/device:CPU:0*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
RestoreV2/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
r
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
	RestoreV2	RestoreV2ConstRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
B
IdentityIdentity	RestoreV2*
T0*
_output_shapes
:
I
AssignVariableOpAssignVariableOpConv1/kernelIdentity*
dtype0
�
RestoreV2_1/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_1	RestoreV2ConstRestoreV2_1/tensor_namesRestoreV2_1/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_1IdentityRestoreV2_1*
T0*
_output_shapes
:
O
AssignVariableOp_1AssignVariableOpbn_Conv1/gamma
Identity_1*
dtype0
�
RestoreV2_2/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_2	RestoreV2ConstRestoreV2_2/tensor_namesRestoreV2_2/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_2IdentityRestoreV2_2*
T0*
_output_shapes
:
N
AssignVariableOp_2AssignVariableOpbn_Conv1/beta
Identity_2*
dtype0
�
RestoreV2_3/tensor_namesConst"/device:CPU:0*P
valueGBEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_3/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_3	RestoreV2ConstRestoreV2_3/tensor_namesRestoreV2_3/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_3IdentityRestoreV2_3*
T0*
_output_shapes
:
U
AssignVariableOp_3AssignVariableOpbn_Conv1/moving_mean
Identity_3*
dtype0
�
RestoreV2_4/tensor_namesConst"/device:CPU:0*T
valueKBIB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_4/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_4	RestoreV2ConstRestoreV2_4/tensor_namesRestoreV2_4/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_4IdentityRestoreV2_4*
T0*
_output_shapes
:
Y
AssignVariableOp_4AssignVariableOpbn_Conv1/moving_variance
Identity_4*
dtype0
�
RestoreV2_5/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_5/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_5	RestoreV2ConstRestoreV2_5/tensor_namesRestoreV2_5/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_5IdentityRestoreV2_5*
T0*
_output_shapes
:
i
AssignVariableOp_5AssignVariableOp(expanded_conv_depthwise/depthwise_kernel
Identity_5*
dtype0
�
RestoreV2_6/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_6/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_6	RestoreV2ConstRestoreV2_6/tensor_namesRestoreV2_6/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_6IdentityRestoreV2_6*
T0*
_output_shapes
:
a
AssignVariableOp_6AssignVariableOp expanded_conv_depthwise_BN/gamma
Identity_6*
dtype0
�
RestoreV2_7/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_7/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_7	RestoreV2ConstRestoreV2_7/tensor_namesRestoreV2_7/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_7IdentityRestoreV2_7*
T0*
_output_shapes
:
`
AssignVariableOp_7AssignVariableOpexpanded_conv_depthwise_BN/beta
Identity_7*
dtype0
�
RestoreV2_8/tensor_namesConst"/device:CPU:0*P
valueGBEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_8/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_8	RestoreV2ConstRestoreV2_8/tensor_namesRestoreV2_8/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_8IdentityRestoreV2_8*
T0*
_output_shapes
:
g
AssignVariableOp_8AssignVariableOp&expanded_conv_depthwise_BN/moving_mean
Identity_8*
dtype0
�
RestoreV2_9/tensor_namesConst"/device:CPU:0*T
valueKBIB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_9/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_9	RestoreV2ConstRestoreV2_9/tensor_namesRestoreV2_9/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_9IdentityRestoreV2_9*
T0*
_output_shapes
:
k
AssignVariableOp_9AssignVariableOp*expanded_conv_depthwise_BN/moving_variance
Identity_9*
dtype0
�
RestoreV2_10/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_10/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_10	RestoreV2ConstRestoreV2_10/tensor_namesRestoreV2_10/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_10IdentityRestoreV2_10*
T0*
_output_shapes
:
_
AssignVariableOp_10AssignVariableOpexpanded_conv_project/kernelIdentity_10*
dtype0
�
RestoreV2_11/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_11/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_11	RestoreV2ConstRestoreV2_11/tensor_namesRestoreV2_11/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_11IdentityRestoreV2_11*
T0*
_output_shapes
:
a
AssignVariableOp_11AssignVariableOpexpanded_conv_project_BN/gammaIdentity_11*
dtype0
�
RestoreV2_12/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_12/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_12	RestoreV2ConstRestoreV2_12/tensor_namesRestoreV2_12/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_12IdentityRestoreV2_12*
T0*
_output_shapes
:
`
AssignVariableOp_12AssignVariableOpexpanded_conv_project_BN/betaIdentity_12*
dtype0
�
RestoreV2_13/tensor_namesConst"/device:CPU:0*P
valueGBEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_13/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_13	RestoreV2ConstRestoreV2_13/tensor_namesRestoreV2_13/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_13IdentityRestoreV2_13*
T0*
_output_shapes
:
g
AssignVariableOp_13AssignVariableOp$expanded_conv_project_BN/moving_meanIdentity_13*
dtype0
�
RestoreV2_14/tensor_namesConst"/device:CPU:0*T
valueKBIB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_14/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_14	RestoreV2ConstRestoreV2_14/tensor_namesRestoreV2_14/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_14IdentityRestoreV2_14*
T0*
_output_shapes
:
k
AssignVariableOp_14AssignVariableOp(expanded_conv_project_BN/moving_varianceIdentity_14*
dtype0
�
RestoreV2_15/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_15/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_15	RestoreV2ConstRestoreV2_15/tensor_namesRestoreV2_15/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_15IdentityRestoreV2_15*
T0*
_output_shapes
:
X
AssignVariableOp_15AssignVariableOpblock_1_expand/kernelIdentity_15*
dtype0
�
RestoreV2_16/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_16/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_16	RestoreV2ConstRestoreV2_16/tensor_namesRestoreV2_16/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_16IdentityRestoreV2_16*
T0*
_output_shapes
:
Z
AssignVariableOp_16AssignVariableOpblock_1_expand_BN/gammaIdentity_16*
dtype0
�
RestoreV2_17/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_17/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_17	RestoreV2ConstRestoreV2_17/tensor_namesRestoreV2_17/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_17IdentityRestoreV2_17*
T0*
_output_shapes
:
Y
AssignVariableOp_17AssignVariableOpblock_1_expand_BN/betaIdentity_17*
dtype0
�
RestoreV2_18/tensor_namesConst"/device:CPU:0*P
valueGBEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_18/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_18	RestoreV2ConstRestoreV2_18/tensor_namesRestoreV2_18/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_18IdentityRestoreV2_18*
T0*
_output_shapes
:
`
AssignVariableOp_18AssignVariableOpblock_1_expand_BN/moving_meanIdentity_18*
dtype0
�
RestoreV2_19/tensor_namesConst"/device:CPU:0*T
valueKBIB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_19/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_19	RestoreV2ConstRestoreV2_19/tensor_namesRestoreV2_19/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_19IdentityRestoreV2_19*
T0*
_output_shapes
:
d
AssignVariableOp_19AssignVariableOp!block_1_expand_BN/moving_varianceIdentity_19*
dtype0
�
RestoreV2_20/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_20/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_20	RestoreV2ConstRestoreV2_20/tensor_namesRestoreV2_20/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_20IdentityRestoreV2_20*
T0*
_output_shapes
:
e
AssignVariableOp_20AssignVariableOp"block_1_depthwise/depthwise_kernelIdentity_20*
dtype0
�
RestoreV2_21/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_21/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_21	RestoreV2ConstRestoreV2_21/tensor_namesRestoreV2_21/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_21IdentityRestoreV2_21*
T0*
_output_shapes
:
]
AssignVariableOp_21AssignVariableOpblock_1_depthwise_BN/gammaIdentity_21*
dtype0
�
RestoreV2_22/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_22/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_22	RestoreV2ConstRestoreV2_22/tensor_namesRestoreV2_22/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_22IdentityRestoreV2_22*
T0*
_output_shapes
:
\
AssignVariableOp_22AssignVariableOpblock_1_depthwise_BN/betaIdentity_22*
dtype0
�
RestoreV2_23/tensor_namesConst"/device:CPU:0*P
valueGBEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_23/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_23	RestoreV2ConstRestoreV2_23/tensor_namesRestoreV2_23/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_23IdentityRestoreV2_23*
T0*
_output_shapes
:
c
AssignVariableOp_23AssignVariableOp block_1_depthwise_BN/moving_meanIdentity_23*
dtype0
�
RestoreV2_24/tensor_namesConst"/device:CPU:0*T
valueKBIB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_24/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_24	RestoreV2ConstRestoreV2_24/tensor_namesRestoreV2_24/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_24IdentityRestoreV2_24*
T0*
_output_shapes
:
g
AssignVariableOp_24AssignVariableOp$block_1_depthwise_BN/moving_varianceIdentity_24*
dtype0
�
RestoreV2_25/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_25/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_25	RestoreV2ConstRestoreV2_25/tensor_namesRestoreV2_25/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_25IdentityRestoreV2_25*
T0*
_output_shapes
:
Y
AssignVariableOp_25AssignVariableOpblock_1_project/kernelIdentity_25*
dtype0
�
RestoreV2_26/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_26/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_26	RestoreV2ConstRestoreV2_26/tensor_namesRestoreV2_26/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_26IdentityRestoreV2_26*
T0*
_output_shapes
:
[
AssignVariableOp_26AssignVariableOpblock_1_project_BN/gammaIdentity_26*
dtype0
�
RestoreV2_27/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_27/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_27	RestoreV2ConstRestoreV2_27/tensor_namesRestoreV2_27/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_27IdentityRestoreV2_27*
T0*
_output_shapes
:
Z
AssignVariableOp_27AssignVariableOpblock_1_project_BN/betaIdentity_27*
dtype0
�
RestoreV2_28/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_28/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_28	RestoreV2ConstRestoreV2_28/tensor_namesRestoreV2_28/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_28IdentityRestoreV2_28*
T0*
_output_shapes
:
a
AssignVariableOp_28AssignVariableOpblock_1_project_BN/moving_meanIdentity_28*
dtype0
�
RestoreV2_29/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_29/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_29	RestoreV2ConstRestoreV2_29/tensor_namesRestoreV2_29/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_29IdentityRestoreV2_29*
T0*
_output_shapes
:
e
AssignVariableOp_29AssignVariableOp"block_1_project_BN/moving_varianceIdentity_29*
dtype0
�
RestoreV2_30/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_30/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_30	RestoreV2ConstRestoreV2_30/tensor_namesRestoreV2_30/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_30IdentityRestoreV2_30*
T0*
_output_shapes
:
X
AssignVariableOp_30AssignVariableOpblock_2_expand/kernelIdentity_30*
dtype0
�
RestoreV2_31/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_31/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_31	RestoreV2ConstRestoreV2_31/tensor_namesRestoreV2_31/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_31IdentityRestoreV2_31*
T0*
_output_shapes
:
Z
AssignVariableOp_31AssignVariableOpblock_2_expand_BN/gammaIdentity_31*
dtype0
�
RestoreV2_32/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_32/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_32	RestoreV2ConstRestoreV2_32/tensor_namesRestoreV2_32/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_32IdentityRestoreV2_32*
T0*
_output_shapes
:
Y
AssignVariableOp_32AssignVariableOpblock_2_expand_BN/betaIdentity_32*
dtype0
�
RestoreV2_33/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_33/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_33	RestoreV2ConstRestoreV2_33/tensor_namesRestoreV2_33/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_33IdentityRestoreV2_33*
T0*
_output_shapes
:
`
AssignVariableOp_33AssignVariableOpblock_2_expand_BN/moving_meanIdentity_33*
dtype0
�
RestoreV2_34/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_34/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_34	RestoreV2ConstRestoreV2_34/tensor_namesRestoreV2_34/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_34IdentityRestoreV2_34*
T0*
_output_shapes
:
d
AssignVariableOp_34AssignVariableOp!block_2_expand_BN/moving_varianceIdentity_34*
dtype0
�
RestoreV2_35/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_35/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_35	RestoreV2ConstRestoreV2_35/tensor_namesRestoreV2_35/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_35IdentityRestoreV2_35*
T0*
_output_shapes
:
e
AssignVariableOp_35AssignVariableOp"block_2_depthwise/depthwise_kernelIdentity_35*
dtype0
�
RestoreV2_36/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_36/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_36	RestoreV2ConstRestoreV2_36/tensor_namesRestoreV2_36/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_36IdentityRestoreV2_36*
T0*
_output_shapes
:
]
AssignVariableOp_36AssignVariableOpblock_2_depthwise_BN/gammaIdentity_36*
dtype0
�
RestoreV2_37/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_37/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_37	RestoreV2ConstRestoreV2_37/tensor_namesRestoreV2_37/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_37IdentityRestoreV2_37*
T0*
_output_shapes
:
\
AssignVariableOp_37AssignVariableOpblock_2_depthwise_BN/betaIdentity_37*
dtype0
�
RestoreV2_38/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_38/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_38	RestoreV2ConstRestoreV2_38/tensor_namesRestoreV2_38/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_38IdentityRestoreV2_38*
T0*
_output_shapes
:
c
AssignVariableOp_38AssignVariableOp block_2_depthwise_BN/moving_meanIdentity_38*
dtype0
�
RestoreV2_39/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_39/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_39	RestoreV2ConstRestoreV2_39/tensor_namesRestoreV2_39/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_39IdentityRestoreV2_39*
T0*
_output_shapes
:
g
AssignVariableOp_39AssignVariableOp$block_2_depthwise_BN/moving_varianceIdentity_39*
dtype0
�
RestoreV2_40/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_40/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_40	RestoreV2ConstRestoreV2_40/tensor_namesRestoreV2_40/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_40IdentityRestoreV2_40*
T0*
_output_shapes
:
Y
AssignVariableOp_40AssignVariableOpblock_2_project/kernelIdentity_40*
dtype0
�
RestoreV2_41/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_41/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_41	RestoreV2ConstRestoreV2_41/tensor_namesRestoreV2_41/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_41IdentityRestoreV2_41*
T0*
_output_shapes
:
[
AssignVariableOp_41AssignVariableOpblock_2_project_BN/gammaIdentity_41*
dtype0
�
RestoreV2_42/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_42/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_42	RestoreV2ConstRestoreV2_42/tensor_namesRestoreV2_42/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_42IdentityRestoreV2_42*
T0*
_output_shapes
:
Z
AssignVariableOp_42AssignVariableOpblock_2_project_BN/betaIdentity_42*
dtype0
�
RestoreV2_43/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_43/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_43	RestoreV2ConstRestoreV2_43/tensor_namesRestoreV2_43/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_43IdentityRestoreV2_43*
T0*
_output_shapes
:
a
AssignVariableOp_43AssignVariableOpblock_2_project_BN/moving_meanIdentity_43*
dtype0
�
RestoreV2_44/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_44/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_44	RestoreV2ConstRestoreV2_44/tensor_namesRestoreV2_44/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_44IdentityRestoreV2_44*
T0*
_output_shapes
:
e
AssignVariableOp_44AssignVariableOp"block_2_project_BN/moving_varianceIdentity_44*
dtype0
�
RestoreV2_45/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_45/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_45	RestoreV2ConstRestoreV2_45/tensor_namesRestoreV2_45/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_45IdentityRestoreV2_45*
T0*
_output_shapes
:
X
AssignVariableOp_45AssignVariableOpblock_3_expand/kernelIdentity_45*
dtype0
�
RestoreV2_46/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_46/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_46	RestoreV2ConstRestoreV2_46/tensor_namesRestoreV2_46/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_46IdentityRestoreV2_46*
T0*
_output_shapes
:
Z
AssignVariableOp_46AssignVariableOpblock_3_expand_BN/gammaIdentity_46*
dtype0
�
RestoreV2_47/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_47/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_47	RestoreV2ConstRestoreV2_47/tensor_namesRestoreV2_47/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_47IdentityRestoreV2_47*
T0*
_output_shapes
:
Y
AssignVariableOp_47AssignVariableOpblock_3_expand_BN/betaIdentity_47*
dtype0
�
RestoreV2_48/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_48/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_48	RestoreV2ConstRestoreV2_48/tensor_namesRestoreV2_48/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_48IdentityRestoreV2_48*
T0*
_output_shapes
:
`
AssignVariableOp_48AssignVariableOpblock_3_expand_BN/moving_meanIdentity_48*
dtype0
�
RestoreV2_49/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_49/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_49	RestoreV2ConstRestoreV2_49/tensor_namesRestoreV2_49/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_49IdentityRestoreV2_49*
T0*
_output_shapes
:
d
AssignVariableOp_49AssignVariableOp!block_3_expand_BN/moving_varianceIdentity_49*
dtype0
�
RestoreV2_50/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-20/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_50/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_50	RestoreV2ConstRestoreV2_50/tensor_namesRestoreV2_50/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_50IdentityRestoreV2_50*
T0*
_output_shapes
:
e
AssignVariableOp_50AssignVariableOp"block_3_depthwise/depthwise_kernelIdentity_50*
dtype0
�
RestoreV2_51/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_51/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_51	RestoreV2ConstRestoreV2_51/tensor_namesRestoreV2_51/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_51IdentityRestoreV2_51*
T0*
_output_shapes
:
]
AssignVariableOp_51AssignVariableOpblock_3_depthwise_BN/gammaIdentity_51*
dtype0
�
RestoreV2_52/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_52/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_52	RestoreV2ConstRestoreV2_52/tensor_namesRestoreV2_52/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_52IdentityRestoreV2_52*
T0*
_output_shapes
:
\
AssignVariableOp_52AssignVariableOpblock_3_depthwise_BN/betaIdentity_52*
dtype0
�
RestoreV2_53/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_53/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_53	RestoreV2ConstRestoreV2_53/tensor_namesRestoreV2_53/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_53IdentityRestoreV2_53*
T0*
_output_shapes
:
c
AssignVariableOp_53AssignVariableOp block_3_depthwise_BN/moving_meanIdentity_53*
dtype0
�
RestoreV2_54/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_54/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_54	RestoreV2ConstRestoreV2_54/tensor_namesRestoreV2_54/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_54IdentityRestoreV2_54*
T0*
_output_shapes
:
g
AssignVariableOp_54AssignVariableOp$block_3_depthwise_BN/moving_varianceIdentity_54*
dtype0
�
RestoreV2_55/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_55/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_55	RestoreV2ConstRestoreV2_55/tensor_namesRestoreV2_55/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_55IdentityRestoreV2_55*
T0*
_output_shapes
:
Y
AssignVariableOp_55AssignVariableOpblock_3_project/kernelIdentity_55*
dtype0
�
RestoreV2_56/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-23/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_56/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_56	RestoreV2ConstRestoreV2_56/tensor_namesRestoreV2_56/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_56IdentityRestoreV2_56*
T0*
_output_shapes
:
[
AssignVariableOp_56AssignVariableOpblock_3_project_BN/gammaIdentity_56*
dtype0
�
RestoreV2_57/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-23/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_57/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_57	RestoreV2ConstRestoreV2_57/tensor_namesRestoreV2_57/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_57IdentityRestoreV2_57*
T0*
_output_shapes
:
Z
AssignVariableOp_57AssignVariableOpblock_3_project_BN/betaIdentity_57*
dtype0
�
RestoreV2_58/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-23/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_58/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_58	RestoreV2ConstRestoreV2_58/tensor_namesRestoreV2_58/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_58IdentityRestoreV2_58*
T0*
_output_shapes
:
a
AssignVariableOp_58AssignVariableOpblock_3_project_BN/moving_meanIdentity_58*
dtype0
�
RestoreV2_59/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-23/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_59/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_59	RestoreV2ConstRestoreV2_59/tensor_namesRestoreV2_59/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_59IdentityRestoreV2_59*
T0*
_output_shapes
:
e
AssignVariableOp_59AssignVariableOp"block_3_project_BN/moving_varianceIdentity_59*
dtype0
�
RestoreV2_60/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_60/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_60	RestoreV2ConstRestoreV2_60/tensor_namesRestoreV2_60/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_60IdentityRestoreV2_60*
T0*
_output_shapes
:
X
AssignVariableOp_60AssignVariableOpblock_4_expand/kernelIdentity_60*
dtype0
�
RestoreV2_61/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-25/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_61/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_61	RestoreV2ConstRestoreV2_61/tensor_namesRestoreV2_61/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_61IdentityRestoreV2_61*
T0*
_output_shapes
:
Z
AssignVariableOp_61AssignVariableOpblock_4_expand_BN/gammaIdentity_61*
dtype0
�
RestoreV2_62/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-25/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_62/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_62	RestoreV2ConstRestoreV2_62/tensor_namesRestoreV2_62/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_62IdentityRestoreV2_62*
T0*
_output_shapes
:
Y
AssignVariableOp_62AssignVariableOpblock_4_expand_BN/betaIdentity_62*
dtype0
�
RestoreV2_63/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-25/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_63/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_63	RestoreV2ConstRestoreV2_63/tensor_namesRestoreV2_63/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_63IdentityRestoreV2_63*
T0*
_output_shapes
:
`
AssignVariableOp_63AssignVariableOpblock_4_expand_BN/moving_meanIdentity_63*
dtype0
�
RestoreV2_64/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-25/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_64/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_64	RestoreV2ConstRestoreV2_64/tensor_namesRestoreV2_64/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_64IdentityRestoreV2_64*
T0*
_output_shapes
:
d
AssignVariableOp_64AssignVariableOp!block_4_expand_BN/moving_varianceIdentity_64*
dtype0
�
RestoreV2_65/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-26/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_65/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_65	RestoreV2ConstRestoreV2_65/tensor_namesRestoreV2_65/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_65IdentityRestoreV2_65*
T0*
_output_shapes
:
e
AssignVariableOp_65AssignVariableOp"block_4_depthwise/depthwise_kernelIdentity_65*
dtype0
�
RestoreV2_66/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-27/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_66/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_66	RestoreV2ConstRestoreV2_66/tensor_namesRestoreV2_66/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_66IdentityRestoreV2_66*
T0*
_output_shapes
:
]
AssignVariableOp_66AssignVariableOpblock_4_depthwise_BN/gammaIdentity_66*
dtype0
�
RestoreV2_67/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-27/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_67/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_67	RestoreV2ConstRestoreV2_67/tensor_namesRestoreV2_67/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_67IdentityRestoreV2_67*
T0*
_output_shapes
:
\
AssignVariableOp_67AssignVariableOpblock_4_depthwise_BN/betaIdentity_67*
dtype0
�
RestoreV2_68/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-27/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_68/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_68	RestoreV2ConstRestoreV2_68/tensor_namesRestoreV2_68/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_68IdentityRestoreV2_68*
T0*
_output_shapes
:
c
AssignVariableOp_68AssignVariableOp block_4_depthwise_BN/moving_meanIdentity_68*
dtype0
�
RestoreV2_69/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-27/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_69/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_69	RestoreV2ConstRestoreV2_69/tensor_namesRestoreV2_69/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_69IdentityRestoreV2_69*
T0*
_output_shapes
:
g
AssignVariableOp_69AssignVariableOp$block_4_depthwise_BN/moving_varianceIdentity_69*
dtype0
�
RestoreV2_70/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_70/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_70	RestoreV2ConstRestoreV2_70/tensor_namesRestoreV2_70/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_70IdentityRestoreV2_70*
T0*
_output_shapes
:
Y
AssignVariableOp_70AssignVariableOpblock_4_project/kernelIdentity_70*
dtype0
�
RestoreV2_71/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-29/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_71/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_71	RestoreV2ConstRestoreV2_71/tensor_namesRestoreV2_71/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_71IdentityRestoreV2_71*
T0*
_output_shapes
:
[
AssignVariableOp_71AssignVariableOpblock_4_project_BN/gammaIdentity_71*
dtype0
�
RestoreV2_72/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-29/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_72/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_72	RestoreV2ConstRestoreV2_72/tensor_namesRestoreV2_72/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_72IdentityRestoreV2_72*
T0*
_output_shapes
:
Z
AssignVariableOp_72AssignVariableOpblock_4_project_BN/betaIdentity_72*
dtype0
�
RestoreV2_73/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-29/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_73/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_73	RestoreV2ConstRestoreV2_73/tensor_namesRestoreV2_73/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_73IdentityRestoreV2_73*
T0*
_output_shapes
:
a
AssignVariableOp_73AssignVariableOpblock_4_project_BN/moving_meanIdentity_73*
dtype0
�
RestoreV2_74/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-29/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_74/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_74	RestoreV2ConstRestoreV2_74/tensor_namesRestoreV2_74/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_74IdentityRestoreV2_74*
T0*
_output_shapes
:
e
AssignVariableOp_74AssignVariableOp"block_4_project_BN/moving_varianceIdentity_74*
dtype0
�
RestoreV2_75/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_75/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_75	RestoreV2ConstRestoreV2_75/tensor_namesRestoreV2_75/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_75IdentityRestoreV2_75*
T0*
_output_shapes
:
X
AssignVariableOp_75AssignVariableOpblock_5_expand/kernelIdentity_75*
dtype0
�
RestoreV2_76/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-31/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_76/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_76	RestoreV2ConstRestoreV2_76/tensor_namesRestoreV2_76/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_76IdentityRestoreV2_76*
T0*
_output_shapes
:
Z
AssignVariableOp_76AssignVariableOpblock_5_expand_BN/gammaIdentity_76*
dtype0
�
RestoreV2_77/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-31/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_77/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_77	RestoreV2ConstRestoreV2_77/tensor_namesRestoreV2_77/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_77IdentityRestoreV2_77*
T0*
_output_shapes
:
Y
AssignVariableOp_77AssignVariableOpblock_5_expand_BN/betaIdentity_77*
dtype0
�
RestoreV2_78/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-31/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_78/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_78	RestoreV2ConstRestoreV2_78/tensor_namesRestoreV2_78/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_78IdentityRestoreV2_78*
T0*
_output_shapes
:
`
AssignVariableOp_78AssignVariableOpblock_5_expand_BN/moving_meanIdentity_78*
dtype0
�
RestoreV2_79/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-31/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_79/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_79	RestoreV2ConstRestoreV2_79/tensor_namesRestoreV2_79/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_79IdentityRestoreV2_79*
T0*
_output_shapes
:
d
AssignVariableOp_79AssignVariableOp!block_5_expand_BN/moving_varianceIdentity_79*
dtype0
�
RestoreV2_80/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-32/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_80/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_80	RestoreV2ConstRestoreV2_80/tensor_namesRestoreV2_80/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_80IdentityRestoreV2_80*
T0*
_output_shapes
:
e
AssignVariableOp_80AssignVariableOp"block_5_depthwise/depthwise_kernelIdentity_80*
dtype0
�
RestoreV2_81/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-33/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_81/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_81	RestoreV2ConstRestoreV2_81/tensor_namesRestoreV2_81/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_81IdentityRestoreV2_81*
T0*
_output_shapes
:
]
AssignVariableOp_81AssignVariableOpblock_5_depthwise_BN/gammaIdentity_81*
dtype0
�
RestoreV2_82/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-33/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_82/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_82	RestoreV2ConstRestoreV2_82/tensor_namesRestoreV2_82/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_82IdentityRestoreV2_82*
T0*
_output_shapes
:
\
AssignVariableOp_82AssignVariableOpblock_5_depthwise_BN/betaIdentity_82*
dtype0
�
RestoreV2_83/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-33/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_83/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_83	RestoreV2ConstRestoreV2_83/tensor_namesRestoreV2_83/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_83IdentityRestoreV2_83*
T0*
_output_shapes
:
c
AssignVariableOp_83AssignVariableOp block_5_depthwise_BN/moving_meanIdentity_83*
dtype0
�
RestoreV2_84/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-33/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_84/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_84	RestoreV2ConstRestoreV2_84/tensor_namesRestoreV2_84/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_84IdentityRestoreV2_84*
T0*
_output_shapes
:
g
AssignVariableOp_84AssignVariableOp$block_5_depthwise_BN/moving_varianceIdentity_84*
dtype0
�
RestoreV2_85/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_85/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_85	RestoreV2ConstRestoreV2_85/tensor_namesRestoreV2_85/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_85IdentityRestoreV2_85*
T0*
_output_shapes
:
Y
AssignVariableOp_85AssignVariableOpblock_5_project/kernelIdentity_85*
dtype0
�
RestoreV2_86/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-35/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_86/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_86	RestoreV2ConstRestoreV2_86/tensor_namesRestoreV2_86/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_86IdentityRestoreV2_86*
T0*
_output_shapes
:
[
AssignVariableOp_86AssignVariableOpblock_5_project_BN/gammaIdentity_86*
dtype0
�
RestoreV2_87/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-35/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_87/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_87	RestoreV2ConstRestoreV2_87/tensor_namesRestoreV2_87/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_87IdentityRestoreV2_87*
T0*
_output_shapes
:
Z
AssignVariableOp_87AssignVariableOpblock_5_project_BN/betaIdentity_87*
dtype0
�
RestoreV2_88/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-35/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_88/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_88	RestoreV2ConstRestoreV2_88/tensor_namesRestoreV2_88/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_88IdentityRestoreV2_88*
T0*
_output_shapes
:
a
AssignVariableOp_88AssignVariableOpblock_5_project_BN/moving_meanIdentity_88*
dtype0
�
RestoreV2_89/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-35/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_89/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_89	RestoreV2ConstRestoreV2_89/tensor_namesRestoreV2_89/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_89IdentityRestoreV2_89*
T0*
_output_shapes
:
e
AssignVariableOp_89AssignVariableOp"block_5_project_BN/moving_varianceIdentity_89*
dtype0
�
RestoreV2_90/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_90/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_90	RestoreV2ConstRestoreV2_90/tensor_namesRestoreV2_90/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_90IdentityRestoreV2_90*
T0*
_output_shapes
:
X
AssignVariableOp_90AssignVariableOpblock_6_expand/kernelIdentity_90*
dtype0
�
RestoreV2_91/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-37/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_91/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_91	RestoreV2ConstRestoreV2_91/tensor_namesRestoreV2_91/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_91IdentityRestoreV2_91*
T0*
_output_shapes
:
Z
AssignVariableOp_91AssignVariableOpblock_6_expand_BN/gammaIdentity_91*
dtype0
�
RestoreV2_92/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-37/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_92/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_92	RestoreV2ConstRestoreV2_92/tensor_namesRestoreV2_92/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_92IdentityRestoreV2_92*
T0*
_output_shapes
:
Y
AssignVariableOp_92AssignVariableOpblock_6_expand_BN/betaIdentity_92*
dtype0
�
RestoreV2_93/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-37/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_93/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_93	RestoreV2ConstRestoreV2_93/tensor_namesRestoreV2_93/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_93IdentityRestoreV2_93*
T0*
_output_shapes
:
`
AssignVariableOp_93AssignVariableOpblock_6_expand_BN/moving_meanIdentity_93*
dtype0
�
RestoreV2_94/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-37/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_94/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_94	RestoreV2ConstRestoreV2_94/tensor_namesRestoreV2_94/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_94IdentityRestoreV2_94*
T0*
_output_shapes
:
d
AssignVariableOp_94AssignVariableOp!block_6_expand_BN/moving_varianceIdentity_94*
dtype0
�
RestoreV2_95/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-38/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_95/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_95	RestoreV2ConstRestoreV2_95/tensor_namesRestoreV2_95/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_95IdentityRestoreV2_95*
T0*
_output_shapes
:
e
AssignVariableOp_95AssignVariableOp"block_6_depthwise/depthwise_kernelIdentity_95*
dtype0
�
RestoreV2_96/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-39/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_96/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_96	RestoreV2ConstRestoreV2_96/tensor_namesRestoreV2_96/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_96IdentityRestoreV2_96*
T0*
_output_shapes
:
]
AssignVariableOp_96AssignVariableOpblock_6_depthwise_BN/gammaIdentity_96*
dtype0
�
RestoreV2_97/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-39/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_97/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_97	RestoreV2ConstRestoreV2_97/tensor_namesRestoreV2_97/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_97IdentityRestoreV2_97*
T0*
_output_shapes
:
\
AssignVariableOp_97AssignVariableOpblock_6_depthwise_BN/betaIdentity_97*
dtype0
�
RestoreV2_98/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-39/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_98/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_98	RestoreV2ConstRestoreV2_98/tensor_namesRestoreV2_98/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_98IdentityRestoreV2_98*
T0*
_output_shapes
:
c
AssignVariableOp_98AssignVariableOp block_6_depthwise_BN/moving_meanIdentity_98*
dtype0
�
RestoreV2_99/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-39/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_99/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_99	RestoreV2ConstRestoreV2_99/tensor_namesRestoreV2_99/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_99IdentityRestoreV2_99*
T0*
_output_shapes
:
g
AssignVariableOp_99AssignVariableOp$block_6_depthwise_BN/moving_varianceIdentity_99*
dtype0
�
RestoreV2_100/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_100/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_100	RestoreV2ConstRestoreV2_100/tensor_namesRestoreV2_100/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_100IdentityRestoreV2_100*
T0*
_output_shapes
:
[
AssignVariableOp_100AssignVariableOpblock_6_project/kernelIdentity_100*
dtype0
�
RestoreV2_101/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-41/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_101/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_101	RestoreV2ConstRestoreV2_101/tensor_namesRestoreV2_101/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_101IdentityRestoreV2_101*
T0*
_output_shapes
:
]
AssignVariableOp_101AssignVariableOpblock_6_project_BN/gammaIdentity_101*
dtype0
�
RestoreV2_102/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-41/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_102/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_102	RestoreV2ConstRestoreV2_102/tensor_namesRestoreV2_102/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_102IdentityRestoreV2_102*
T0*
_output_shapes
:
\
AssignVariableOp_102AssignVariableOpblock_6_project_BN/betaIdentity_102*
dtype0
�
RestoreV2_103/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-41/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_103/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_103	RestoreV2ConstRestoreV2_103/tensor_namesRestoreV2_103/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_103IdentityRestoreV2_103*
T0*
_output_shapes
:
c
AssignVariableOp_103AssignVariableOpblock_6_project_BN/moving_meanIdentity_103*
dtype0
�
RestoreV2_104/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-41/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_104/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_104	RestoreV2ConstRestoreV2_104/tensor_namesRestoreV2_104/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_104IdentityRestoreV2_104*
T0*
_output_shapes
:
g
AssignVariableOp_104AssignVariableOp"block_6_project_BN/moving_varianceIdentity_104*
dtype0
�
RestoreV2_105/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-42/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_105/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_105	RestoreV2ConstRestoreV2_105/tensor_namesRestoreV2_105/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_105IdentityRestoreV2_105*
T0*
_output_shapes
:
Z
AssignVariableOp_105AssignVariableOpblock_7_expand/kernelIdentity_105*
dtype0
�
RestoreV2_106/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-43/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_106/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_106	RestoreV2ConstRestoreV2_106/tensor_namesRestoreV2_106/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_106IdentityRestoreV2_106*
T0*
_output_shapes
:
\
AssignVariableOp_106AssignVariableOpblock_7_expand_BN/gammaIdentity_106*
dtype0
�
RestoreV2_107/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-43/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_107/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_107	RestoreV2ConstRestoreV2_107/tensor_namesRestoreV2_107/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_107IdentityRestoreV2_107*
T0*
_output_shapes
:
[
AssignVariableOp_107AssignVariableOpblock_7_expand_BN/betaIdentity_107*
dtype0
�
RestoreV2_108/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-43/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_108/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_108	RestoreV2ConstRestoreV2_108/tensor_namesRestoreV2_108/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_108IdentityRestoreV2_108*
T0*
_output_shapes
:
b
AssignVariableOp_108AssignVariableOpblock_7_expand_BN/moving_meanIdentity_108*
dtype0
�
RestoreV2_109/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-43/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_109/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_109	RestoreV2ConstRestoreV2_109/tensor_namesRestoreV2_109/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_109IdentityRestoreV2_109*
T0*
_output_shapes
:
f
AssignVariableOp_109AssignVariableOp!block_7_expand_BN/moving_varianceIdentity_109*
dtype0
�
RestoreV2_110/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-44/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_110/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_110	RestoreV2ConstRestoreV2_110/tensor_namesRestoreV2_110/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_110IdentityRestoreV2_110*
T0*
_output_shapes
:
g
AssignVariableOp_110AssignVariableOp"block_7_depthwise/depthwise_kernelIdentity_110*
dtype0
�
RestoreV2_111/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-45/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_111/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_111	RestoreV2ConstRestoreV2_111/tensor_namesRestoreV2_111/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_111IdentityRestoreV2_111*
T0*
_output_shapes
:
_
AssignVariableOp_111AssignVariableOpblock_7_depthwise_BN/gammaIdentity_111*
dtype0
�
RestoreV2_112/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-45/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_112/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_112	RestoreV2ConstRestoreV2_112/tensor_namesRestoreV2_112/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_112IdentityRestoreV2_112*
T0*
_output_shapes
:
^
AssignVariableOp_112AssignVariableOpblock_7_depthwise_BN/betaIdentity_112*
dtype0
�
RestoreV2_113/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-45/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_113/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_113	RestoreV2ConstRestoreV2_113/tensor_namesRestoreV2_113/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_113IdentityRestoreV2_113*
T0*
_output_shapes
:
e
AssignVariableOp_113AssignVariableOp block_7_depthwise_BN/moving_meanIdentity_113*
dtype0
�
RestoreV2_114/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-45/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_114/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_114	RestoreV2ConstRestoreV2_114/tensor_namesRestoreV2_114/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_114IdentityRestoreV2_114*
T0*
_output_shapes
:
i
AssignVariableOp_114AssignVariableOp$block_7_depthwise_BN/moving_varianceIdentity_114*
dtype0
�
RestoreV2_115/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-46/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_115/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_115	RestoreV2ConstRestoreV2_115/tensor_namesRestoreV2_115/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_115IdentityRestoreV2_115*
T0*
_output_shapes
:
[
AssignVariableOp_115AssignVariableOpblock_7_project/kernelIdentity_115*
dtype0
�
RestoreV2_116/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-47/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_116/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_116	RestoreV2ConstRestoreV2_116/tensor_namesRestoreV2_116/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_116IdentityRestoreV2_116*
T0*
_output_shapes
:
]
AssignVariableOp_116AssignVariableOpblock_7_project_BN/gammaIdentity_116*
dtype0
�
RestoreV2_117/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-47/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_117/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_117	RestoreV2ConstRestoreV2_117/tensor_namesRestoreV2_117/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_117IdentityRestoreV2_117*
T0*
_output_shapes
:
\
AssignVariableOp_117AssignVariableOpblock_7_project_BN/betaIdentity_117*
dtype0
�
RestoreV2_118/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-47/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_118/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_118	RestoreV2ConstRestoreV2_118/tensor_namesRestoreV2_118/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_118IdentityRestoreV2_118*
T0*
_output_shapes
:
c
AssignVariableOp_118AssignVariableOpblock_7_project_BN/moving_meanIdentity_118*
dtype0
�
RestoreV2_119/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-47/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_119/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_119	RestoreV2ConstRestoreV2_119/tensor_namesRestoreV2_119/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_119IdentityRestoreV2_119*
T0*
_output_shapes
:
g
AssignVariableOp_119AssignVariableOp"block_7_project_BN/moving_varianceIdentity_119*
dtype0
�
RestoreV2_120/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-48/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_120/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_120	RestoreV2ConstRestoreV2_120/tensor_namesRestoreV2_120/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_120IdentityRestoreV2_120*
T0*
_output_shapes
:
Z
AssignVariableOp_120AssignVariableOpblock_8_expand/kernelIdentity_120*
dtype0
�
RestoreV2_121/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-49/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_121/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_121	RestoreV2ConstRestoreV2_121/tensor_namesRestoreV2_121/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_121IdentityRestoreV2_121*
T0*
_output_shapes
:
\
AssignVariableOp_121AssignVariableOpblock_8_expand_BN/gammaIdentity_121*
dtype0
�
RestoreV2_122/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-49/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_122/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_122	RestoreV2ConstRestoreV2_122/tensor_namesRestoreV2_122/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_122IdentityRestoreV2_122*
T0*
_output_shapes
:
[
AssignVariableOp_122AssignVariableOpblock_8_expand_BN/betaIdentity_122*
dtype0
�
RestoreV2_123/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-49/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_123/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_123	RestoreV2ConstRestoreV2_123/tensor_namesRestoreV2_123/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_123IdentityRestoreV2_123*
T0*
_output_shapes
:
b
AssignVariableOp_123AssignVariableOpblock_8_expand_BN/moving_meanIdentity_123*
dtype0
�
RestoreV2_124/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-49/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_124/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_124	RestoreV2ConstRestoreV2_124/tensor_namesRestoreV2_124/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_124IdentityRestoreV2_124*
T0*
_output_shapes
:
f
AssignVariableOp_124AssignVariableOp!block_8_expand_BN/moving_varianceIdentity_124*
dtype0
�
RestoreV2_125/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-50/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_125/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_125	RestoreV2ConstRestoreV2_125/tensor_namesRestoreV2_125/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_125IdentityRestoreV2_125*
T0*
_output_shapes
:
g
AssignVariableOp_125AssignVariableOp"block_8_depthwise/depthwise_kernelIdentity_125*
dtype0
�
RestoreV2_126/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-51/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_126/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_126	RestoreV2ConstRestoreV2_126/tensor_namesRestoreV2_126/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_126IdentityRestoreV2_126*
T0*
_output_shapes
:
_
AssignVariableOp_126AssignVariableOpblock_8_depthwise_BN/gammaIdentity_126*
dtype0
�
RestoreV2_127/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-51/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_127/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_127	RestoreV2ConstRestoreV2_127/tensor_namesRestoreV2_127/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_127IdentityRestoreV2_127*
T0*
_output_shapes
:
^
AssignVariableOp_127AssignVariableOpblock_8_depthwise_BN/betaIdentity_127*
dtype0
�
RestoreV2_128/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-51/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_128/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_128	RestoreV2ConstRestoreV2_128/tensor_namesRestoreV2_128/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_128IdentityRestoreV2_128*
T0*
_output_shapes
:
e
AssignVariableOp_128AssignVariableOp block_8_depthwise_BN/moving_meanIdentity_128*
dtype0
�
RestoreV2_129/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-51/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_129/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_129	RestoreV2ConstRestoreV2_129/tensor_namesRestoreV2_129/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_129IdentityRestoreV2_129*
T0*
_output_shapes
:
i
AssignVariableOp_129AssignVariableOp$block_8_depthwise_BN/moving_varianceIdentity_129*
dtype0
�
RestoreV2_130/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-52/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_130/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_130	RestoreV2ConstRestoreV2_130/tensor_namesRestoreV2_130/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_130IdentityRestoreV2_130*
T0*
_output_shapes
:
[
AssignVariableOp_130AssignVariableOpblock_8_project/kernelIdentity_130*
dtype0
�
RestoreV2_131/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-53/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_131/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_131	RestoreV2ConstRestoreV2_131/tensor_namesRestoreV2_131/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_131IdentityRestoreV2_131*
T0*
_output_shapes
:
]
AssignVariableOp_131AssignVariableOpblock_8_project_BN/gammaIdentity_131*
dtype0
�
RestoreV2_132/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-53/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_132/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_132	RestoreV2ConstRestoreV2_132/tensor_namesRestoreV2_132/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_132IdentityRestoreV2_132*
T0*
_output_shapes
:
\
AssignVariableOp_132AssignVariableOpblock_8_project_BN/betaIdentity_132*
dtype0
�
RestoreV2_133/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-53/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_133/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_133	RestoreV2ConstRestoreV2_133/tensor_namesRestoreV2_133/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_133IdentityRestoreV2_133*
T0*
_output_shapes
:
c
AssignVariableOp_133AssignVariableOpblock_8_project_BN/moving_meanIdentity_133*
dtype0
�
RestoreV2_134/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-53/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_134/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_134	RestoreV2ConstRestoreV2_134/tensor_namesRestoreV2_134/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_134IdentityRestoreV2_134*
T0*
_output_shapes
:
g
AssignVariableOp_134AssignVariableOp"block_8_project_BN/moving_varianceIdentity_134*
dtype0
�
RestoreV2_135/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-54/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_135/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_135	RestoreV2ConstRestoreV2_135/tensor_namesRestoreV2_135/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_135IdentityRestoreV2_135*
T0*
_output_shapes
:
Z
AssignVariableOp_135AssignVariableOpblock_9_expand/kernelIdentity_135*
dtype0
�
RestoreV2_136/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-55/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_136/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_136	RestoreV2ConstRestoreV2_136/tensor_namesRestoreV2_136/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_136IdentityRestoreV2_136*
T0*
_output_shapes
:
\
AssignVariableOp_136AssignVariableOpblock_9_expand_BN/gammaIdentity_136*
dtype0
�
RestoreV2_137/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-55/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_137/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_137	RestoreV2ConstRestoreV2_137/tensor_namesRestoreV2_137/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_137IdentityRestoreV2_137*
T0*
_output_shapes
:
[
AssignVariableOp_137AssignVariableOpblock_9_expand_BN/betaIdentity_137*
dtype0
�
RestoreV2_138/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-55/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_138/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_138	RestoreV2ConstRestoreV2_138/tensor_namesRestoreV2_138/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_138IdentityRestoreV2_138*
T0*
_output_shapes
:
b
AssignVariableOp_138AssignVariableOpblock_9_expand_BN/moving_meanIdentity_138*
dtype0
�
RestoreV2_139/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-55/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_139/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_139	RestoreV2ConstRestoreV2_139/tensor_namesRestoreV2_139/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_139IdentityRestoreV2_139*
T0*
_output_shapes
:
f
AssignVariableOp_139AssignVariableOp!block_9_expand_BN/moving_varianceIdentity_139*
dtype0
�
RestoreV2_140/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-56/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_140/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_140	RestoreV2ConstRestoreV2_140/tensor_namesRestoreV2_140/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_140IdentityRestoreV2_140*
T0*
_output_shapes
:
g
AssignVariableOp_140AssignVariableOp"block_9_depthwise/depthwise_kernelIdentity_140*
dtype0
�
RestoreV2_141/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-57/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_141/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_141	RestoreV2ConstRestoreV2_141/tensor_namesRestoreV2_141/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_141IdentityRestoreV2_141*
T0*
_output_shapes
:
_
AssignVariableOp_141AssignVariableOpblock_9_depthwise_BN/gammaIdentity_141*
dtype0
�
RestoreV2_142/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-57/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_142/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_142	RestoreV2ConstRestoreV2_142/tensor_namesRestoreV2_142/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_142IdentityRestoreV2_142*
T0*
_output_shapes
:
^
AssignVariableOp_142AssignVariableOpblock_9_depthwise_BN/betaIdentity_142*
dtype0
�
RestoreV2_143/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-57/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_143/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_143	RestoreV2ConstRestoreV2_143/tensor_namesRestoreV2_143/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_143IdentityRestoreV2_143*
T0*
_output_shapes
:
e
AssignVariableOp_143AssignVariableOp block_9_depthwise_BN/moving_meanIdentity_143*
dtype0
�
RestoreV2_144/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-57/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_144/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_144	RestoreV2ConstRestoreV2_144/tensor_namesRestoreV2_144/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_144IdentityRestoreV2_144*
T0*
_output_shapes
:
i
AssignVariableOp_144AssignVariableOp$block_9_depthwise_BN/moving_varianceIdentity_144*
dtype0
�
RestoreV2_145/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-58/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_145/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_145	RestoreV2ConstRestoreV2_145/tensor_namesRestoreV2_145/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_145IdentityRestoreV2_145*
T0*
_output_shapes
:
[
AssignVariableOp_145AssignVariableOpblock_9_project/kernelIdentity_145*
dtype0
�
RestoreV2_146/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-59/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_146/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_146	RestoreV2ConstRestoreV2_146/tensor_namesRestoreV2_146/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_146IdentityRestoreV2_146*
T0*
_output_shapes
:
]
AssignVariableOp_146AssignVariableOpblock_9_project_BN/gammaIdentity_146*
dtype0
�
RestoreV2_147/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-59/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_147/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_147	RestoreV2ConstRestoreV2_147/tensor_namesRestoreV2_147/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_147IdentityRestoreV2_147*
T0*
_output_shapes
:
\
AssignVariableOp_147AssignVariableOpblock_9_project_BN/betaIdentity_147*
dtype0
�
RestoreV2_148/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-59/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_148/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_148	RestoreV2ConstRestoreV2_148/tensor_namesRestoreV2_148/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_148IdentityRestoreV2_148*
T0*
_output_shapes
:
c
AssignVariableOp_148AssignVariableOpblock_9_project_BN/moving_meanIdentity_148*
dtype0
�
RestoreV2_149/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-59/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_149/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_149	RestoreV2ConstRestoreV2_149/tensor_namesRestoreV2_149/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_149IdentityRestoreV2_149*
T0*
_output_shapes
:
g
AssignVariableOp_149AssignVariableOp"block_9_project_BN/moving_varianceIdentity_149*
dtype0
�
RestoreV2_150/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-60/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_150/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_150	RestoreV2ConstRestoreV2_150/tensor_namesRestoreV2_150/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_150IdentityRestoreV2_150*
T0*
_output_shapes
:
[
AssignVariableOp_150AssignVariableOpblock_10_expand/kernelIdentity_150*
dtype0
�
RestoreV2_151/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-61/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_151/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_151	RestoreV2ConstRestoreV2_151/tensor_namesRestoreV2_151/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_151IdentityRestoreV2_151*
T0*
_output_shapes
:
]
AssignVariableOp_151AssignVariableOpblock_10_expand_BN/gammaIdentity_151*
dtype0
�
RestoreV2_152/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-61/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_152/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_152	RestoreV2ConstRestoreV2_152/tensor_namesRestoreV2_152/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_152IdentityRestoreV2_152*
T0*
_output_shapes
:
\
AssignVariableOp_152AssignVariableOpblock_10_expand_BN/betaIdentity_152*
dtype0
�
RestoreV2_153/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-61/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_153/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_153	RestoreV2ConstRestoreV2_153/tensor_namesRestoreV2_153/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_153IdentityRestoreV2_153*
T0*
_output_shapes
:
c
AssignVariableOp_153AssignVariableOpblock_10_expand_BN/moving_meanIdentity_153*
dtype0
�
RestoreV2_154/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-61/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_154/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_154	RestoreV2ConstRestoreV2_154/tensor_namesRestoreV2_154/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_154IdentityRestoreV2_154*
T0*
_output_shapes
:
g
AssignVariableOp_154AssignVariableOp"block_10_expand_BN/moving_varianceIdentity_154*
dtype0
�
RestoreV2_155/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-62/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_155/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_155	RestoreV2ConstRestoreV2_155/tensor_namesRestoreV2_155/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_155IdentityRestoreV2_155*
T0*
_output_shapes
:
h
AssignVariableOp_155AssignVariableOp#block_10_depthwise/depthwise_kernelIdentity_155*
dtype0
�
RestoreV2_156/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-63/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_156/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_156	RestoreV2ConstRestoreV2_156/tensor_namesRestoreV2_156/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_156IdentityRestoreV2_156*
T0*
_output_shapes
:
`
AssignVariableOp_156AssignVariableOpblock_10_depthwise_BN/gammaIdentity_156*
dtype0
�
RestoreV2_157/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-63/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_157/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_157	RestoreV2ConstRestoreV2_157/tensor_namesRestoreV2_157/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_157IdentityRestoreV2_157*
T0*
_output_shapes
:
_
AssignVariableOp_157AssignVariableOpblock_10_depthwise_BN/betaIdentity_157*
dtype0
�
RestoreV2_158/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-63/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_158/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_158	RestoreV2ConstRestoreV2_158/tensor_namesRestoreV2_158/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_158IdentityRestoreV2_158*
T0*
_output_shapes
:
f
AssignVariableOp_158AssignVariableOp!block_10_depthwise_BN/moving_meanIdentity_158*
dtype0
�
RestoreV2_159/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-63/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_159/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_159	RestoreV2ConstRestoreV2_159/tensor_namesRestoreV2_159/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_159IdentityRestoreV2_159*
T0*
_output_shapes
:
j
AssignVariableOp_159AssignVariableOp%block_10_depthwise_BN/moving_varianceIdentity_159*
dtype0
�
RestoreV2_160/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-64/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_160/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_160	RestoreV2ConstRestoreV2_160/tensor_namesRestoreV2_160/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_160IdentityRestoreV2_160*
T0*
_output_shapes
:
\
AssignVariableOp_160AssignVariableOpblock_10_project/kernelIdentity_160*
dtype0
�
RestoreV2_161/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-65/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_161/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_161	RestoreV2ConstRestoreV2_161/tensor_namesRestoreV2_161/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_161IdentityRestoreV2_161*
T0*
_output_shapes
:
^
AssignVariableOp_161AssignVariableOpblock_10_project_BN/gammaIdentity_161*
dtype0
�
RestoreV2_162/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-65/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_162/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_162	RestoreV2ConstRestoreV2_162/tensor_namesRestoreV2_162/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_162IdentityRestoreV2_162*
T0*
_output_shapes
:
]
AssignVariableOp_162AssignVariableOpblock_10_project_BN/betaIdentity_162*
dtype0
�
RestoreV2_163/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-65/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_163/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_163	RestoreV2ConstRestoreV2_163/tensor_namesRestoreV2_163/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_163IdentityRestoreV2_163*
T0*
_output_shapes
:
d
AssignVariableOp_163AssignVariableOpblock_10_project_BN/moving_meanIdentity_163*
dtype0
�
RestoreV2_164/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-65/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_164/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_164	RestoreV2ConstRestoreV2_164/tensor_namesRestoreV2_164/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_164IdentityRestoreV2_164*
T0*
_output_shapes
:
h
AssignVariableOp_164AssignVariableOp#block_10_project_BN/moving_varianceIdentity_164*
dtype0
�
RestoreV2_165/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-66/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_165/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_165	RestoreV2ConstRestoreV2_165/tensor_namesRestoreV2_165/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_165IdentityRestoreV2_165*
T0*
_output_shapes
:
[
AssignVariableOp_165AssignVariableOpblock_11_expand/kernelIdentity_165*
dtype0
�
RestoreV2_166/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-67/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_166/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_166	RestoreV2ConstRestoreV2_166/tensor_namesRestoreV2_166/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_166IdentityRestoreV2_166*
T0*
_output_shapes
:
]
AssignVariableOp_166AssignVariableOpblock_11_expand_BN/gammaIdentity_166*
dtype0
�
RestoreV2_167/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-67/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_167/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_167	RestoreV2ConstRestoreV2_167/tensor_namesRestoreV2_167/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_167IdentityRestoreV2_167*
T0*
_output_shapes
:
\
AssignVariableOp_167AssignVariableOpblock_11_expand_BN/betaIdentity_167*
dtype0
�
RestoreV2_168/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-67/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_168/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_168	RestoreV2ConstRestoreV2_168/tensor_namesRestoreV2_168/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_168IdentityRestoreV2_168*
T0*
_output_shapes
:
c
AssignVariableOp_168AssignVariableOpblock_11_expand_BN/moving_meanIdentity_168*
dtype0
�
RestoreV2_169/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-67/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_169/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_169	RestoreV2ConstRestoreV2_169/tensor_namesRestoreV2_169/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_169IdentityRestoreV2_169*
T0*
_output_shapes
:
g
AssignVariableOp_169AssignVariableOp"block_11_expand_BN/moving_varianceIdentity_169*
dtype0
�
RestoreV2_170/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-68/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_170/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_170	RestoreV2ConstRestoreV2_170/tensor_namesRestoreV2_170/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_170IdentityRestoreV2_170*
T0*
_output_shapes
:
h
AssignVariableOp_170AssignVariableOp#block_11_depthwise/depthwise_kernelIdentity_170*
dtype0
�
RestoreV2_171/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-69/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_171/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_171	RestoreV2ConstRestoreV2_171/tensor_namesRestoreV2_171/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_171IdentityRestoreV2_171*
T0*
_output_shapes
:
`
AssignVariableOp_171AssignVariableOpblock_11_depthwise_BN/gammaIdentity_171*
dtype0
�
RestoreV2_172/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-69/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_172/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_172	RestoreV2ConstRestoreV2_172/tensor_namesRestoreV2_172/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_172IdentityRestoreV2_172*
T0*
_output_shapes
:
_
AssignVariableOp_172AssignVariableOpblock_11_depthwise_BN/betaIdentity_172*
dtype0
�
RestoreV2_173/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-69/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_173/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_173	RestoreV2ConstRestoreV2_173/tensor_namesRestoreV2_173/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_173IdentityRestoreV2_173*
T0*
_output_shapes
:
f
AssignVariableOp_173AssignVariableOp!block_11_depthwise_BN/moving_meanIdentity_173*
dtype0
�
RestoreV2_174/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-69/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_174/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_174	RestoreV2ConstRestoreV2_174/tensor_namesRestoreV2_174/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_174IdentityRestoreV2_174*
T0*
_output_shapes
:
j
AssignVariableOp_174AssignVariableOp%block_11_depthwise_BN/moving_varianceIdentity_174*
dtype0
�
RestoreV2_175/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-70/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_175/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_175	RestoreV2ConstRestoreV2_175/tensor_namesRestoreV2_175/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_175IdentityRestoreV2_175*
T0*
_output_shapes
:
\
AssignVariableOp_175AssignVariableOpblock_11_project/kernelIdentity_175*
dtype0
�
RestoreV2_176/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-71/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_176/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_176	RestoreV2ConstRestoreV2_176/tensor_namesRestoreV2_176/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_176IdentityRestoreV2_176*
T0*
_output_shapes
:
^
AssignVariableOp_176AssignVariableOpblock_11_project_BN/gammaIdentity_176*
dtype0
�
RestoreV2_177/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-71/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_177/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_177	RestoreV2ConstRestoreV2_177/tensor_namesRestoreV2_177/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_177IdentityRestoreV2_177*
T0*
_output_shapes
:
]
AssignVariableOp_177AssignVariableOpblock_11_project_BN/betaIdentity_177*
dtype0
�
RestoreV2_178/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-71/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_178/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_178	RestoreV2ConstRestoreV2_178/tensor_namesRestoreV2_178/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_178IdentityRestoreV2_178*
T0*
_output_shapes
:
d
AssignVariableOp_178AssignVariableOpblock_11_project_BN/moving_meanIdentity_178*
dtype0
�
RestoreV2_179/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-71/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_179/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_179	RestoreV2ConstRestoreV2_179/tensor_namesRestoreV2_179/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_179IdentityRestoreV2_179*
T0*
_output_shapes
:
h
AssignVariableOp_179AssignVariableOp#block_11_project_BN/moving_varianceIdentity_179*
dtype0
�
RestoreV2_180/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-72/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_180/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_180	RestoreV2ConstRestoreV2_180/tensor_namesRestoreV2_180/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_180IdentityRestoreV2_180*
T0*
_output_shapes
:
[
AssignVariableOp_180AssignVariableOpblock_12_expand/kernelIdentity_180*
dtype0
�
RestoreV2_181/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-73/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_181/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_181	RestoreV2ConstRestoreV2_181/tensor_namesRestoreV2_181/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_181IdentityRestoreV2_181*
T0*
_output_shapes
:
]
AssignVariableOp_181AssignVariableOpblock_12_expand_BN/gammaIdentity_181*
dtype0
�
RestoreV2_182/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-73/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_182/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_182	RestoreV2ConstRestoreV2_182/tensor_namesRestoreV2_182/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_182IdentityRestoreV2_182*
T0*
_output_shapes
:
\
AssignVariableOp_182AssignVariableOpblock_12_expand_BN/betaIdentity_182*
dtype0
�
RestoreV2_183/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-73/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_183/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_183	RestoreV2ConstRestoreV2_183/tensor_namesRestoreV2_183/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_183IdentityRestoreV2_183*
T0*
_output_shapes
:
c
AssignVariableOp_183AssignVariableOpblock_12_expand_BN/moving_meanIdentity_183*
dtype0
�
RestoreV2_184/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-73/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_184/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_184	RestoreV2ConstRestoreV2_184/tensor_namesRestoreV2_184/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_184IdentityRestoreV2_184*
T0*
_output_shapes
:
g
AssignVariableOp_184AssignVariableOp"block_12_expand_BN/moving_varianceIdentity_184*
dtype0
�
RestoreV2_185/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-74/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_185/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_185	RestoreV2ConstRestoreV2_185/tensor_namesRestoreV2_185/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_185IdentityRestoreV2_185*
T0*
_output_shapes
:
h
AssignVariableOp_185AssignVariableOp#block_12_depthwise/depthwise_kernelIdentity_185*
dtype0
�
RestoreV2_186/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-75/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_186/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_186	RestoreV2ConstRestoreV2_186/tensor_namesRestoreV2_186/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_186IdentityRestoreV2_186*
T0*
_output_shapes
:
`
AssignVariableOp_186AssignVariableOpblock_12_depthwise_BN/gammaIdentity_186*
dtype0
�
RestoreV2_187/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-75/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_187/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_187	RestoreV2ConstRestoreV2_187/tensor_namesRestoreV2_187/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_187IdentityRestoreV2_187*
T0*
_output_shapes
:
_
AssignVariableOp_187AssignVariableOpblock_12_depthwise_BN/betaIdentity_187*
dtype0
�
RestoreV2_188/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-75/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_188/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_188	RestoreV2ConstRestoreV2_188/tensor_namesRestoreV2_188/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_188IdentityRestoreV2_188*
T0*
_output_shapes
:
f
AssignVariableOp_188AssignVariableOp!block_12_depthwise_BN/moving_meanIdentity_188*
dtype0
�
RestoreV2_189/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-75/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_189/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_189	RestoreV2ConstRestoreV2_189/tensor_namesRestoreV2_189/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_189IdentityRestoreV2_189*
T0*
_output_shapes
:
j
AssignVariableOp_189AssignVariableOp%block_12_depthwise_BN/moving_varianceIdentity_189*
dtype0
�
RestoreV2_190/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-76/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_190/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_190	RestoreV2ConstRestoreV2_190/tensor_namesRestoreV2_190/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_190IdentityRestoreV2_190*
T0*
_output_shapes
:
\
AssignVariableOp_190AssignVariableOpblock_12_project/kernelIdentity_190*
dtype0
�
RestoreV2_191/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-77/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_191/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_191	RestoreV2ConstRestoreV2_191/tensor_namesRestoreV2_191/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_191IdentityRestoreV2_191*
T0*
_output_shapes
:
^
AssignVariableOp_191AssignVariableOpblock_12_project_BN/gammaIdentity_191*
dtype0
�
RestoreV2_192/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-77/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_192/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_192	RestoreV2ConstRestoreV2_192/tensor_namesRestoreV2_192/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_192IdentityRestoreV2_192*
T0*
_output_shapes
:
]
AssignVariableOp_192AssignVariableOpblock_12_project_BN/betaIdentity_192*
dtype0
�
RestoreV2_193/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-77/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_193/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_193	RestoreV2ConstRestoreV2_193/tensor_namesRestoreV2_193/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_193IdentityRestoreV2_193*
T0*
_output_shapes
:
d
AssignVariableOp_193AssignVariableOpblock_12_project_BN/moving_meanIdentity_193*
dtype0
�
RestoreV2_194/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-77/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_194/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_194	RestoreV2ConstRestoreV2_194/tensor_namesRestoreV2_194/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_194IdentityRestoreV2_194*
T0*
_output_shapes
:
h
AssignVariableOp_194AssignVariableOp#block_12_project_BN/moving_varianceIdentity_194*
dtype0
�
RestoreV2_195/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-78/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_195/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_195	RestoreV2ConstRestoreV2_195/tensor_namesRestoreV2_195/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_195IdentityRestoreV2_195*
T0*
_output_shapes
:
[
AssignVariableOp_195AssignVariableOpblock_13_expand/kernelIdentity_195*
dtype0
�
RestoreV2_196/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-79/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_196/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_196	RestoreV2ConstRestoreV2_196/tensor_namesRestoreV2_196/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_196IdentityRestoreV2_196*
T0*
_output_shapes
:
]
AssignVariableOp_196AssignVariableOpblock_13_expand_BN/gammaIdentity_196*
dtype0
�
RestoreV2_197/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-79/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_197/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_197	RestoreV2ConstRestoreV2_197/tensor_namesRestoreV2_197/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_197IdentityRestoreV2_197*
T0*
_output_shapes
:
\
AssignVariableOp_197AssignVariableOpblock_13_expand_BN/betaIdentity_197*
dtype0
�
RestoreV2_198/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-79/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_198/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_198	RestoreV2ConstRestoreV2_198/tensor_namesRestoreV2_198/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_198IdentityRestoreV2_198*
T0*
_output_shapes
:
c
AssignVariableOp_198AssignVariableOpblock_13_expand_BN/moving_meanIdentity_198*
dtype0
�
RestoreV2_199/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-79/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_199/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_199	RestoreV2ConstRestoreV2_199/tensor_namesRestoreV2_199/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_199IdentityRestoreV2_199*
T0*
_output_shapes
:
g
AssignVariableOp_199AssignVariableOp"block_13_expand_BN/moving_varianceIdentity_199*
dtype0
�
RestoreV2_200/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-80/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_200/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_200	RestoreV2ConstRestoreV2_200/tensor_namesRestoreV2_200/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_200IdentityRestoreV2_200*
T0*
_output_shapes
:
h
AssignVariableOp_200AssignVariableOp#block_13_depthwise/depthwise_kernelIdentity_200*
dtype0
�
RestoreV2_201/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-81/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_201/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_201	RestoreV2ConstRestoreV2_201/tensor_namesRestoreV2_201/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_201IdentityRestoreV2_201*
T0*
_output_shapes
:
`
AssignVariableOp_201AssignVariableOpblock_13_depthwise_BN/gammaIdentity_201*
dtype0
�
RestoreV2_202/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-81/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_202/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_202	RestoreV2ConstRestoreV2_202/tensor_namesRestoreV2_202/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_202IdentityRestoreV2_202*
T0*
_output_shapes
:
_
AssignVariableOp_202AssignVariableOpblock_13_depthwise_BN/betaIdentity_202*
dtype0
�
RestoreV2_203/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-81/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_203/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_203	RestoreV2ConstRestoreV2_203/tensor_namesRestoreV2_203/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_203IdentityRestoreV2_203*
T0*
_output_shapes
:
f
AssignVariableOp_203AssignVariableOp!block_13_depthwise_BN/moving_meanIdentity_203*
dtype0
�
RestoreV2_204/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-81/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_204/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_204	RestoreV2ConstRestoreV2_204/tensor_namesRestoreV2_204/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_204IdentityRestoreV2_204*
T0*
_output_shapes
:
j
AssignVariableOp_204AssignVariableOp%block_13_depthwise_BN/moving_varianceIdentity_204*
dtype0
�
RestoreV2_205/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-82/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_205/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_205	RestoreV2ConstRestoreV2_205/tensor_namesRestoreV2_205/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_205IdentityRestoreV2_205*
T0*
_output_shapes
:
\
AssignVariableOp_205AssignVariableOpblock_13_project/kernelIdentity_205*
dtype0
�
RestoreV2_206/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-83/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_206/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_206	RestoreV2ConstRestoreV2_206/tensor_namesRestoreV2_206/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_206IdentityRestoreV2_206*
T0*
_output_shapes
:
^
AssignVariableOp_206AssignVariableOpblock_13_project_BN/gammaIdentity_206*
dtype0
�
RestoreV2_207/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-83/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_207/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_207	RestoreV2ConstRestoreV2_207/tensor_namesRestoreV2_207/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_207IdentityRestoreV2_207*
T0*
_output_shapes
:
]
AssignVariableOp_207AssignVariableOpblock_13_project_BN/betaIdentity_207*
dtype0
�
RestoreV2_208/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-83/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_208/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_208	RestoreV2ConstRestoreV2_208/tensor_namesRestoreV2_208/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_208IdentityRestoreV2_208*
T0*
_output_shapes
:
d
AssignVariableOp_208AssignVariableOpblock_13_project_BN/moving_meanIdentity_208*
dtype0
�
RestoreV2_209/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-83/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_209/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_209	RestoreV2ConstRestoreV2_209/tensor_namesRestoreV2_209/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_209IdentityRestoreV2_209*
T0*
_output_shapes
:
h
AssignVariableOp_209AssignVariableOp#block_13_project_BN/moving_varianceIdentity_209*
dtype0
�
RestoreV2_210/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-84/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_210/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_210	RestoreV2ConstRestoreV2_210/tensor_namesRestoreV2_210/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_210IdentityRestoreV2_210*
T0*
_output_shapes
:
[
AssignVariableOp_210AssignVariableOpblock_14_expand/kernelIdentity_210*
dtype0
�
RestoreV2_211/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-85/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_211/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_211	RestoreV2ConstRestoreV2_211/tensor_namesRestoreV2_211/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_211IdentityRestoreV2_211*
T0*
_output_shapes
:
]
AssignVariableOp_211AssignVariableOpblock_14_expand_BN/gammaIdentity_211*
dtype0
�
RestoreV2_212/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-85/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_212/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_212	RestoreV2ConstRestoreV2_212/tensor_namesRestoreV2_212/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_212IdentityRestoreV2_212*
T0*
_output_shapes
:
\
AssignVariableOp_212AssignVariableOpblock_14_expand_BN/betaIdentity_212*
dtype0
�
RestoreV2_213/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-85/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_213/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_213	RestoreV2ConstRestoreV2_213/tensor_namesRestoreV2_213/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_213IdentityRestoreV2_213*
T0*
_output_shapes
:
c
AssignVariableOp_213AssignVariableOpblock_14_expand_BN/moving_meanIdentity_213*
dtype0
�
RestoreV2_214/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-85/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_214/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_214	RestoreV2ConstRestoreV2_214/tensor_namesRestoreV2_214/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_214IdentityRestoreV2_214*
T0*
_output_shapes
:
g
AssignVariableOp_214AssignVariableOp"block_14_expand_BN/moving_varianceIdentity_214*
dtype0
�
RestoreV2_215/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-86/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_215/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_215	RestoreV2ConstRestoreV2_215/tensor_namesRestoreV2_215/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_215IdentityRestoreV2_215*
T0*
_output_shapes
:
h
AssignVariableOp_215AssignVariableOp#block_14_depthwise/depthwise_kernelIdentity_215*
dtype0
�
RestoreV2_216/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-87/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_216/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_216	RestoreV2ConstRestoreV2_216/tensor_namesRestoreV2_216/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_216IdentityRestoreV2_216*
T0*
_output_shapes
:
`
AssignVariableOp_216AssignVariableOpblock_14_depthwise_BN/gammaIdentity_216*
dtype0
�
RestoreV2_217/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-87/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_217/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_217	RestoreV2ConstRestoreV2_217/tensor_namesRestoreV2_217/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_217IdentityRestoreV2_217*
T0*
_output_shapes
:
_
AssignVariableOp_217AssignVariableOpblock_14_depthwise_BN/betaIdentity_217*
dtype0
�
RestoreV2_218/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-87/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_218/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_218	RestoreV2ConstRestoreV2_218/tensor_namesRestoreV2_218/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_218IdentityRestoreV2_218*
T0*
_output_shapes
:
f
AssignVariableOp_218AssignVariableOp!block_14_depthwise_BN/moving_meanIdentity_218*
dtype0
�
RestoreV2_219/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-87/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_219/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_219	RestoreV2ConstRestoreV2_219/tensor_namesRestoreV2_219/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_219IdentityRestoreV2_219*
T0*
_output_shapes
:
j
AssignVariableOp_219AssignVariableOp%block_14_depthwise_BN/moving_varianceIdentity_219*
dtype0
�
RestoreV2_220/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-88/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_220/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_220	RestoreV2ConstRestoreV2_220/tensor_namesRestoreV2_220/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_220IdentityRestoreV2_220*
T0*
_output_shapes
:
\
AssignVariableOp_220AssignVariableOpblock_14_project/kernelIdentity_220*
dtype0
�
RestoreV2_221/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-89/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_221/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_221	RestoreV2ConstRestoreV2_221/tensor_namesRestoreV2_221/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_221IdentityRestoreV2_221*
T0*
_output_shapes
:
^
AssignVariableOp_221AssignVariableOpblock_14_project_BN/gammaIdentity_221*
dtype0
�
RestoreV2_222/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-89/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_222/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_222	RestoreV2ConstRestoreV2_222/tensor_namesRestoreV2_222/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_222IdentityRestoreV2_222*
T0*
_output_shapes
:
]
AssignVariableOp_222AssignVariableOpblock_14_project_BN/betaIdentity_222*
dtype0
�
RestoreV2_223/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-89/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_223/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_223	RestoreV2ConstRestoreV2_223/tensor_namesRestoreV2_223/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_223IdentityRestoreV2_223*
T0*
_output_shapes
:
d
AssignVariableOp_223AssignVariableOpblock_14_project_BN/moving_meanIdentity_223*
dtype0
�
RestoreV2_224/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-89/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_224/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_224	RestoreV2ConstRestoreV2_224/tensor_namesRestoreV2_224/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_224IdentityRestoreV2_224*
T0*
_output_shapes
:
h
AssignVariableOp_224AssignVariableOp#block_14_project_BN/moving_varianceIdentity_224*
dtype0
�
RestoreV2_225/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-90/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_225/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_225	RestoreV2ConstRestoreV2_225/tensor_namesRestoreV2_225/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_225IdentityRestoreV2_225*
T0*
_output_shapes
:
[
AssignVariableOp_225AssignVariableOpblock_15_expand/kernelIdentity_225*
dtype0
�
RestoreV2_226/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-91/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_226/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_226	RestoreV2ConstRestoreV2_226/tensor_namesRestoreV2_226/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_226IdentityRestoreV2_226*
T0*
_output_shapes
:
]
AssignVariableOp_226AssignVariableOpblock_15_expand_BN/gammaIdentity_226*
dtype0
�
RestoreV2_227/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-91/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_227/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_227	RestoreV2ConstRestoreV2_227/tensor_namesRestoreV2_227/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_227IdentityRestoreV2_227*
T0*
_output_shapes
:
\
AssignVariableOp_227AssignVariableOpblock_15_expand_BN/betaIdentity_227*
dtype0
�
RestoreV2_228/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-91/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_228/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_228	RestoreV2ConstRestoreV2_228/tensor_namesRestoreV2_228/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_228IdentityRestoreV2_228*
T0*
_output_shapes
:
c
AssignVariableOp_228AssignVariableOpblock_15_expand_BN/moving_meanIdentity_228*
dtype0
�
RestoreV2_229/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-91/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_229/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_229	RestoreV2ConstRestoreV2_229/tensor_namesRestoreV2_229/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_229IdentityRestoreV2_229*
T0*
_output_shapes
:
g
AssignVariableOp_229AssignVariableOp"block_15_expand_BN/moving_varianceIdentity_229*
dtype0
�
RestoreV2_230/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-92/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_230/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_230	RestoreV2ConstRestoreV2_230/tensor_namesRestoreV2_230/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_230IdentityRestoreV2_230*
T0*
_output_shapes
:
h
AssignVariableOp_230AssignVariableOp#block_15_depthwise/depthwise_kernelIdentity_230*
dtype0
�
RestoreV2_231/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-93/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_231/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_231	RestoreV2ConstRestoreV2_231/tensor_namesRestoreV2_231/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_231IdentityRestoreV2_231*
T0*
_output_shapes
:
`
AssignVariableOp_231AssignVariableOpblock_15_depthwise_BN/gammaIdentity_231*
dtype0
�
RestoreV2_232/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-93/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_232/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_232	RestoreV2ConstRestoreV2_232/tensor_namesRestoreV2_232/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_232IdentityRestoreV2_232*
T0*
_output_shapes
:
_
AssignVariableOp_232AssignVariableOpblock_15_depthwise_BN/betaIdentity_232*
dtype0
�
RestoreV2_233/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-93/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_233/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_233	RestoreV2ConstRestoreV2_233/tensor_namesRestoreV2_233/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_233IdentityRestoreV2_233*
T0*
_output_shapes
:
f
AssignVariableOp_233AssignVariableOp!block_15_depthwise_BN/moving_meanIdentity_233*
dtype0
�
RestoreV2_234/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-93/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_234/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_234	RestoreV2ConstRestoreV2_234/tensor_namesRestoreV2_234/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_234IdentityRestoreV2_234*
T0*
_output_shapes
:
j
AssignVariableOp_234AssignVariableOp%block_15_depthwise_BN/moving_varianceIdentity_234*
dtype0
�
RestoreV2_235/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-94/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_235/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_235	RestoreV2ConstRestoreV2_235/tensor_namesRestoreV2_235/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_235IdentityRestoreV2_235*
T0*
_output_shapes
:
\
AssignVariableOp_235AssignVariableOpblock_15_project/kernelIdentity_235*
dtype0
�
RestoreV2_236/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-95/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_236/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_236	RestoreV2ConstRestoreV2_236/tensor_namesRestoreV2_236/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_236IdentityRestoreV2_236*
T0*
_output_shapes
:
^
AssignVariableOp_236AssignVariableOpblock_15_project_BN/gammaIdentity_236*
dtype0
�
RestoreV2_237/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-95/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_237/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_237	RestoreV2ConstRestoreV2_237/tensor_namesRestoreV2_237/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_237IdentityRestoreV2_237*
T0*
_output_shapes
:
]
AssignVariableOp_237AssignVariableOpblock_15_project_BN/betaIdentity_237*
dtype0
�
RestoreV2_238/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-95/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_238/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_238	RestoreV2ConstRestoreV2_238/tensor_namesRestoreV2_238/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_238IdentityRestoreV2_238*
T0*
_output_shapes
:
d
AssignVariableOp_238AssignVariableOpblock_15_project_BN/moving_meanIdentity_238*
dtype0
�
RestoreV2_239/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-95/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_239/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_239	RestoreV2ConstRestoreV2_239/tensor_namesRestoreV2_239/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_239IdentityRestoreV2_239*
T0*
_output_shapes
:
h
AssignVariableOp_239AssignVariableOp#block_15_project_BN/moving_varianceIdentity_239*
dtype0
�
RestoreV2_240/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-96/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_240/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_240	RestoreV2ConstRestoreV2_240/tensor_namesRestoreV2_240/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_240IdentityRestoreV2_240*
T0*
_output_shapes
:
[
AssignVariableOp_240AssignVariableOpblock_16_expand/kernelIdentity_240*
dtype0
�
RestoreV2_241/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-97/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_241/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_241	RestoreV2ConstRestoreV2_241/tensor_namesRestoreV2_241/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_241IdentityRestoreV2_241*
T0*
_output_shapes
:
]
AssignVariableOp_241AssignVariableOpblock_16_expand_BN/gammaIdentity_241*
dtype0
�
RestoreV2_242/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-97/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_242/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_242	RestoreV2ConstRestoreV2_242/tensor_namesRestoreV2_242/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_242IdentityRestoreV2_242*
T0*
_output_shapes
:
\
AssignVariableOp_242AssignVariableOpblock_16_expand_BN/betaIdentity_242*
dtype0
�
RestoreV2_243/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-97/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_243/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_243	RestoreV2ConstRestoreV2_243/tensor_namesRestoreV2_243/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_243IdentityRestoreV2_243*
T0*
_output_shapes
:
c
AssignVariableOp_243AssignVariableOpblock_16_expand_BN/moving_meanIdentity_243*
dtype0
�
RestoreV2_244/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-97/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_244/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_244	RestoreV2ConstRestoreV2_244/tensor_namesRestoreV2_244/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_244IdentityRestoreV2_244*
T0*
_output_shapes
:
g
AssignVariableOp_244AssignVariableOp"block_16_expand_BN/moving_varianceIdentity_244*
dtype0
�
RestoreV2_245/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-98/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_245/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_245	RestoreV2ConstRestoreV2_245/tensor_namesRestoreV2_245/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_245IdentityRestoreV2_245*
T0*
_output_shapes
:
h
AssignVariableOp_245AssignVariableOp#block_16_depthwise/depthwise_kernelIdentity_245*
dtype0
�
RestoreV2_246/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-99/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_246/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_246	RestoreV2ConstRestoreV2_246/tensor_namesRestoreV2_246/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_246IdentityRestoreV2_246*
T0*
_output_shapes
:
`
AssignVariableOp_246AssignVariableOpblock_16_depthwise_BN/gammaIdentity_246*
dtype0
�
RestoreV2_247/tensor_namesConst"/device:CPU:0*J
valueAB?B5layer_with_weights-99/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_247/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_247	RestoreV2ConstRestoreV2_247/tensor_namesRestoreV2_247/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_247IdentityRestoreV2_247*
T0*
_output_shapes
:
_
AssignVariableOp_247AssignVariableOpblock_16_depthwise_BN/betaIdentity_247*
dtype0
�
RestoreV2_248/tensor_namesConst"/device:CPU:0*Q
valueHBFB<layer_with_weights-99/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_248/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_248	RestoreV2ConstRestoreV2_248/tensor_namesRestoreV2_248/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_248IdentityRestoreV2_248*
T0*
_output_shapes
:
f
AssignVariableOp_248AssignVariableOp!block_16_depthwise_BN/moving_meanIdentity_248*
dtype0
�
RestoreV2_249/tensor_namesConst"/device:CPU:0*U
valueLBJB@layer_with_weights-99/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_249/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_249	RestoreV2ConstRestoreV2_249/tensor_namesRestoreV2_249/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_249IdentityRestoreV2_249*
T0*
_output_shapes
:
j
AssignVariableOp_249AssignVariableOp%block_16_depthwise_BN/moving_varianceIdentity_249*
dtype0
�
RestoreV2_250/tensor_namesConst"/device:CPU:0*M
valueDBBB8layer_with_weights-100/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_250/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_250	RestoreV2ConstRestoreV2_250/tensor_namesRestoreV2_250/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_250IdentityRestoreV2_250*
T0*
_output_shapes
:
\
AssignVariableOp_250AssignVariableOpblock_16_project/kernelIdentity_250*
dtype0
�
RestoreV2_251/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-101/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_251/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_251	RestoreV2ConstRestoreV2_251/tensor_namesRestoreV2_251/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_251IdentityRestoreV2_251*
T0*
_output_shapes
:
^
AssignVariableOp_251AssignVariableOpblock_16_project_BN/gammaIdentity_251*
dtype0
�
RestoreV2_252/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-101/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_252/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_252	RestoreV2ConstRestoreV2_252/tensor_namesRestoreV2_252/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_252IdentityRestoreV2_252*
T0*
_output_shapes
:
]
AssignVariableOp_252AssignVariableOpblock_16_project_BN/betaIdentity_252*
dtype0
�
RestoreV2_253/tensor_namesConst"/device:CPU:0*R
valueIBGB=layer_with_weights-101/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_253/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_253	RestoreV2ConstRestoreV2_253/tensor_namesRestoreV2_253/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_253IdentityRestoreV2_253*
T0*
_output_shapes
:
d
AssignVariableOp_253AssignVariableOpblock_16_project_BN/moving_meanIdentity_253*
dtype0
�
RestoreV2_254/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-101/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_254/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_254	RestoreV2ConstRestoreV2_254/tensor_namesRestoreV2_254/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_254IdentityRestoreV2_254*
T0*
_output_shapes
:
h
AssignVariableOp_254AssignVariableOp#block_16_project_BN/moving_varianceIdentity_254*
dtype0
�
RestoreV2_255/tensor_namesConst"/device:CPU:0*M
valueDBBB8layer_with_weights-102/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_255/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_255	RestoreV2ConstRestoreV2_255/tensor_namesRestoreV2_255/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_255IdentityRestoreV2_255*
T0*
_output_shapes
:
R
AssignVariableOp_255AssignVariableOpConv_1/kernelIdentity_255*
dtype0
�
RestoreV2_256/tensor_namesConst"/device:CPU:0*L
valueCBAB7layer_with_weights-103/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_256/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_256	RestoreV2ConstRestoreV2_256/tensor_namesRestoreV2_256/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_256IdentityRestoreV2_256*
T0*
_output_shapes
:
T
AssignVariableOp_256AssignVariableOpConv_1_bn/gammaIdentity_256*
dtype0
�
RestoreV2_257/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-103/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_257/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_257	RestoreV2ConstRestoreV2_257/tensor_namesRestoreV2_257/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_257IdentityRestoreV2_257*
T0*
_output_shapes
:
S
AssignVariableOp_257AssignVariableOpConv_1_bn/betaIdentity_257*
dtype0
�
RestoreV2_258/tensor_namesConst"/device:CPU:0*R
valueIBGB=layer_with_weights-103/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_258/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_258	RestoreV2ConstRestoreV2_258/tensor_namesRestoreV2_258/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_258IdentityRestoreV2_258*
T0*
_output_shapes
:
Z
AssignVariableOp_258AssignVariableOpConv_1_bn/moving_meanIdentity_258*
dtype0
�
RestoreV2_259/tensor_namesConst"/device:CPU:0*V
valueMBKBAlayer_with_weights-103/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
v
RestoreV2_259/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
RestoreV2_259	RestoreV2ConstRestoreV2_259/tensor_namesRestoreV2_259/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
J
Identity_259IdentityRestoreV2_259*
T0*
_output_shapes
:
^
AssignVariableOp_259AssignVariableOpConv_1_bn/moving_varianceIdentity_259*
dtype0
\
VarIsInitializedOpVarIsInitializedOpblock_6_depthwise_BN/gamma*
_output_shapes
: 
b
VarIsInitializedOp_1VarIsInitializedOpblock_10_expand_BN/moving_mean*
_output_shapes
: 
f
VarIsInitializedOp_2VarIsInitializedOp"block_2_project_BN/moving_variance*
_output_shapes
: 
[
VarIsInitializedOp_3VarIsInitializedOpblock_4_expand_BN/gamma*
_output_shapes
: 
e
VarIsInitializedOp_4VarIsInitializedOp!block_6_expand_BN/moving_variance*
_output_shapes
: 
\
VarIsInitializedOp_5VarIsInitializedOpblock_5_project_BN/gamma*
_output_shapes
: 
h
VarIsInitializedOp_6VarIsInitializedOp$block_9_depthwise_BN/moving_variance*
_output_shapes
: 
b
VarIsInitializedOp_7VarIsInitializedOpblock_11_expand_BN/moving_mean*
_output_shapes
: 
]
VarIsInitializedOp_8VarIsInitializedOpblock_11_project_BN/gamma*
_output_shapes
: 
^
VarIsInitializedOp_9VarIsInitializedOpblock_1_depthwise_BN/gamma*
_output_shapes
: 
\
VarIsInitializedOp_10VarIsInitializedOpblock_3_expand_BN/gamma*
_output_shapes
: 
i
VarIsInitializedOp_11VarIsInitializedOp$block_4_depthwise_BN/moving_variance*
_output_shapes
: 
b
VarIsInitializedOp_12VarIsInitializedOpblock_5_expand_BN/moving_mean*
_output_shapes
: 
[
VarIsInitializedOp_13VarIsInitializedOpblock_3_project/kernel*
_output_shapes
: 
g
VarIsInitializedOp_14VarIsInitializedOp"block_7_depthwise/depthwise_kernel*
_output_shapes
: 
\
VarIsInitializedOp_15VarIsInitializedOpblock_9_expand_BN/gamma*
_output_shapes
: 
g
VarIsInitializedOp_16VarIsInitializedOp"block_12_expand_BN/moving_variance*
_output_shapes
: 
_
VarIsInitializedOp_17VarIsInitializedOpblock_13_depthwise_BN/beta*
_output_shapes
: 
_
VarIsInitializedOp_18VarIsInitializedOpblock_16_depthwise_BN/beta*
_output_shapes
: 
^
VarIsInitializedOp_19VarIsInitializedOpblock_1_depthwise_BN/beta*
_output_shapes
: 
`
VarIsInitializedOp_20VarIsInitializedOpblock_10_depthwise_BN/gamma*
_output_shapes
: 
g
VarIsInitializedOp_21VarIsInitializedOp"block_11_expand_BN/moving_variance*
_output_shapes
: 
f
VarIsInitializedOp_22VarIsInitializedOp!block_16_depthwise_BN/moving_mean*
_output_shapes
: 
^
VarIsInitializedOp_23VarIsInitializedOpblock_6_depthwise_BN/beta*
_output_shapes
: 
\
VarIsInitializedOp_24VarIsInitializedOpblock_8_expand_BN/gamma*
_output_shapes
: 
g
VarIsInitializedOp_25VarIsInitializedOp"block_10_expand_BN/moving_variance*
_output_shapes
: 
f
VarIsInitializedOp_26VarIsInitializedOp!block_13_depthwise_BN/moving_mean*
_output_shapes
: 
e
VarIsInitializedOp_27VarIsInitializedOp block_1_depthwise_BN/moving_mean*
_output_shapes
: 
\
VarIsInitializedOp_28VarIsInitializedOpblock_5_project_BN/beta*
_output_shapes
: 
Z
VarIsInitializedOp_29VarIsInitializedOpblock_3_expand/kernel*
_output_shapes
: 
f
VarIsInitializedOp_30VarIsInitializedOp!block_5_expand_BN/moving_variance*
_output_shapes
: 
[
VarIsInitializedOp_31VarIsInitializedOpblock_9_expand_BN/beta*
_output_shapes
: 
c
VarIsInitializedOp_32VarIsInitializedOpexpanded_conv_project_BN/gamma*
_output_shapes
: 
e
VarIsInitializedOp_33VarIsInitializedOp block_6_depthwise_BN/moving_mean*
_output_shapes
: 
]
VarIsInitializedOp_34VarIsInitializedOpblock_11_project_BN/beta*
_output_shapes
: 
_
VarIsInitializedOp_35VarIsInitializedOpblock_15_depthwise_BN/beta*
_output_shapes
: 
[
VarIsInitializedOp_36VarIsInitializedOpblock_4_expand_BN/beta*
_output_shapes
: 
[
VarIsInitializedOp_37VarIsInitializedOpblock_4_project/kernel*
_output_shapes
: 
c
VarIsInitializedOp_38VarIsInitializedOpblock_5_project_BN/moving_mean*
_output_shapes
: 
h
VarIsInitializedOp_39VarIsInitializedOp#block_12_depthwise/depthwise_kernel*
_output_shapes
: 
b
VarIsInitializedOp_40VarIsInitializedOpblock_9_expand_BN/moving_mean*
_output_shapes
: 
]
VarIsInitializedOp_41VarIsInitializedOpblock_16_expand_BN/gamma*
_output_shapes
: 
j
VarIsInitializedOp_42VarIsInitializedOp%block_16_depthwise_BN/moving_variance*
_output_shapes
: 
d
VarIsInitializedOp_43VarIsInitializedOpblock_11_project_BN/moving_mean*
_output_shapes
: 
f
VarIsInitializedOp_44VarIsInitializedOp!block_15_depthwise_BN/moving_mean*
_output_shapes
: 
j
VarIsInitializedOp_45VarIsInitializedOp%block_13_depthwise_BN/moving_variance*
_output_shapes
: 
i
VarIsInitializedOp_46VarIsInitializedOp$block_1_depthwise_BN/moving_variance*
_output_shapes
: 
b
VarIsInitializedOp_47VarIsInitializedOpblock_4_expand_BN/moving_mean*
_output_shapes
: 
[
VarIsInitializedOp_48VarIsInitializedOpblock_3_expand_BN/beta*
_output_shapes
: 
i
VarIsInitializedOp_49VarIsInitializedOp$block_6_depthwise_BN/moving_variance*
_output_shapes
: 
]
VarIsInitializedOp_50VarIsInitializedOpblock_8_project_BN/gamma*
_output_shapes
: 
\
VarIsInitializedOp_51VarIsInitializedOpblock_10_project/kernel*
_output_shapes
: 
a
VarIsInitializedOp_52VarIsInitializedOpexpanded_conv_project/kernel*
_output_shapes
: 
]
VarIsInitializedOp_53VarIsInitializedOpblock_1_project_BN/gamma*
_output_shapes
: 
g
VarIsInitializedOp_54VarIsInitializedOp"block_5_project_BN/moving_variance*
_output_shapes
: 
e
VarIsInitializedOp_55VarIsInitializedOp expanded_conv_depthwise_BN/gamma*
_output_shapes
: 
m
VarIsInitializedOp_56VarIsInitializedOp(expanded_conv_project_BN/moving_variance*
_output_shapes
: 
f
VarIsInitializedOp_57VarIsInitializedOp!block_9_expand_BN/moving_variance*
_output_shapes
: 
d
VarIsInitializedOp_58VarIsInitializedOpblock_10_project_BN/moving_mean*
_output_shapes
: 
b
VarIsInitializedOp_59VarIsInitializedOpblock_3_expand_BN/moving_mean*
_output_shapes
: 
\
VarIsInitializedOp_60VarIsInitializedOpblock_7_expand_BN/gamma*
_output_shapes
: 
_
VarIsInitializedOp_61VarIsInitializedOpblock_10_depthwise_BN/beta*
_output_shapes
: 
h
VarIsInitializedOp_62VarIsInitializedOp#block_11_depthwise/depthwise_kernel*
_output_shapes
: 
]
VarIsInitializedOp_63VarIsInitializedOpblock_6_project_BN/gamma*
_output_shapes
: 
[
VarIsInitializedOp_64VarIsInitializedOpblock_8_expand_BN/beta*
_output_shapes
: 
h
VarIsInitializedOp_65VarIsInitializedOp#block_11_project_BN/moving_variance*
_output_shapes
: 
j
VarIsInitializedOp_66VarIsInitializedOp%block_15_depthwise_BN/moving_variance*
_output_shapes
: 
f
VarIsInitializedOp_67VarIsInitializedOp!block_4_expand_BN/moving_variance*
_output_shapes
: 
\
VarIsInitializedOp_68VarIsInitializedOpblock_16_expand_BN/beta*
_output_shapes
: 
f
VarIsInitializedOp_69VarIsInitializedOp!block_7_expand_BN/moving_variance*
_output_shapes
: 
g
VarIsInitializedOp_70VarIsInitializedOp"block_13_expand_BN/moving_variance*
_output_shapes
: 
[
VarIsInitializedOp_71VarIsInitializedOpblock_1_project/kernel*
_output_shapes
: 
c
VarIsInitializedOp_72VarIsInitializedOpblock_16_expand_BN/moving_mean*
_output_shapes
: 
\
VarIsInitializedOp_73VarIsInitializedOpblock_16_project/kernel*
_output_shapes
: 
b
VarIsInitializedOp_74VarIsInitializedOpblock_8_expand_BN/moving_mean*
_output_shapes
: 
`
VarIsInitializedOp_75VarIsInitializedOpblock_11_depthwise_BN/gamma*
_output_shapes
: 
\
VarIsInitializedOp_76VarIsInitializedOpblock_13_project/kernel*
_output_shapes
: 
h
VarIsInitializedOp_77VarIsInitializedOp#block_16_project_BN/moving_variance*
_output_shapes
: 
\
VarIsInitializedOp_78VarIsInitializedOpblock_6_expand_BN/gamma*
_output_shapes
: 
Z
VarIsInitializedOp_79VarIsInitializedOpblock_7_expand/kernel*
_output_shapes
: 
j
VarIsInitializedOp_80VarIsInitializedOp%block_11_depthwise_BN/moving_variance*
_output_shapes
: 
[
VarIsInitializedOp_81VarIsInitializedOpblock_13_expand/kernel*
_output_shapes
: 
^
VarIsInitializedOp_82VarIsInitializedOpblock_10_project_BN/gamma*
_output_shapes
: 
[
VarIsInitializedOp_83VarIsInitializedOpblock_14_expand/kernel*
_output_shapes
: 
f
VarIsInitializedOp_84VarIsInitializedOp!block_3_expand_BN/moving_variance*
_output_shapes
: 
g
VarIsInitializedOp_85VarIsInitializedOp"block_3_depthwise/depthwise_kernel*
_output_shapes
: 
]
VarIsInitializedOp_86VarIsInitializedOpblock_14_expand_BN/gamma*
_output_shapes
: 
m
VarIsInitializedOp_87VarIsInitializedOp(expanded_conv_depthwise/depthwise_kernel*
_output_shapes
: 
Z
VarIsInitializedOp_88VarIsInitializedOpblock_1_expand/kernel*
_output_shapes
: 
f
VarIsInitializedOp_89VarIsInitializedOp!block_1_expand_BN/moving_variance*
_output_shapes
: 
]
VarIsInitializedOp_90VarIsInitializedOpblock_7_project_BN/gamma*
_output_shapes
: 
[
VarIsInitializedOp_91VarIsInitializedOpblock_2_project/kernel*
_output_shapes
: 
g
VarIsInitializedOp_92VarIsInitializedOp"block_16_expand_BN/moving_variance*
_output_shapes
: 
f
VarIsInitializedOp_93VarIsInitializedOp!block_8_expand_BN/moving_variance*
_output_shapes
: 
\
VarIsInitializedOp_94VarIsInitializedOpblock_8_project_BN/beta*
_output_shapes
: 
\
VarIsInitializedOp_95VarIsInitializedOpblock_15_project/kernel*
_output_shapes
: 
\
VarIsInitializedOp_96VarIsInitializedOpblock_1_project_BN/beta*
_output_shapes
: 
g
VarIsInitializedOp_97VarIsInitializedOp"block_4_depthwise/depthwise_kernel*
_output_shapes
: 
_
VarIsInitializedOp_98VarIsInitializedOpblock_11_depthwise_BN/beta*
_output_shapes
: 
[
VarIsInitializedOp_99VarIsInitializedOpblock_12_expand/kernel*
_output_shapes
: 
g
VarIsInitializedOp_100VarIsInitializedOp!block_10_depthwise_BN/moving_mean*
_output_shapes
: 
_
VarIsInitializedOp_101VarIsInitializedOpblock_15_project_BN/gamma*
_output_shapes
: 
\
VarIsInitializedOp_102VarIsInitializedOpblock_7_expand_BN/beta*
_output_shapes
: 
\
VarIsInitializedOp_103VarIsInitializedOpblock_7_project/kernel*
_output_shapes
: 
d
VarIsInitializedOp_104VarIsInitializedOpblock_8_project_BN/moving_mean*
_output_shapes
: 
]
VarIsInitializedOp_105VarIsInitializedOpblock_6_project_BN/beta*
_output_shapes
: 
^
VarIsInitializedOp_106VarIsInitializedOpblock_10_expand_BN/gamma*
_output_shapes
: 
]
VarIsInitializedOp_107VarIsInitializedOpblock_14_expand_BN/beta*
_output_shapes
: 
c
VarIsInitializedOp_108VarIsInitializedOpexpanded_conv_project_BN/beta*
_output_shapes
: 
d
VarIsInitializedOp_109VarIsInitializedOpblock_1_project_BN/moving_mean*
_output_shapes
: 
g
VarIsInitializedOp_110VarIsInitializedOp!block_11_depthwise_BN/moving_mean*
_output_shapes
: 
`
VarIsInitializedOp_111VarIsInitializedOpblock_4_depthwise_BN/gamma*
_output_shapes
: 
c
VarIsInitializedOp_112VarIsInitializedOpblock_7_expand_BN/moving_mean*
_output_shapes
: 
d
VarIsInitializedOp_113VarIsInitializedOpblock_6_project_BN/moving_mean*
_output_shapes
: 
d
VarIsInitializedOp_114VarIsInitializedOpblock_14_expand_BN/moving_mean*
_output_shapes
: 
]
VarIsInitializedOp_115VarIsInitializedOpblock_14_project/kernel*
_output_shapes
: 
R
VarIsInitializedOp_116VarIsInitializedOpConv1/kernel*
_output_shapes
: 
\
VarIsInitializedOp_117VarIsInitializedOpblock_6_expand_BN/beta*
_output_shapes
: 
\
VarIsInitializedOp_118VarIsInitializedOpblock_11_expand/kernel*
_output_shapes
: 
^
VarIsInitializedOp_119VarIsInitializedOpblock_11_expand_BN/gamma*
_output_shapes
: 
h
VarIsInitializedOp_120VarIsInitializedOp"block_8_project_BN/moving_variance*
_output_shapes
: 
\
VarIsInitializedOp_121VarIsInitializedOpblock_10_expand/kernel*
_output_shapes
: 
^
VarIsInitializedOp_122VarIsInitializedOpblock_15_project_BN/beta*
_output_shapes
: 
i
VarIsInitializedOp_123VarIsInitializedOp#block_10_project_BN/moving_variance*
_output_shapes
: 
j
VarIsInitializedOp_124VarIsInitializedOp$expanded_conv_project_BN/moving_mean*
_output_shapes
: 
h
VarIsInitializedOp_125VarIsInitializedOp"block_1_project_BN/moving_variance*
_output_shapes
: 
c
VarIsInitializedOp_126VarIsInitializedOpblock_6_expand_BN/moving_mean*
_output_shapes
: 
[
VarIsInitializedOp_127VarIsInitializedOpblock_5_expand/kernel*
_output_shapes
: 
]
VarIsInitializedOp_128VarIsInitializedOpblock_7_project_BN/beta*
_output_shapes
: 
e
VarIsInitializedOp_129VarIsInitializedOpblock_15_project_BN/moving_mean*
_output_shapes
: 
_
VarIsInitializedOp_130VarIsInitializedOpblock_4_depthwise_BN/beta*
_output_shapes
: 
i
VarIsInitializedOp_131VarIsInitializedOp#block_13_depthwise/depthwise_kernel*
_output_shapes
: 
h
VarIsInitializedOp_132VarIsInitializedOp"block_6_project_BN/moving_variance*
_output_shapes
: 
d
VarIsInitializedOp_133VarIsInitializedOpblock_7_project_BN/moving_mean*
_output_shapes
: 
`
VarIsInitializedOp_134VarIsInitializedOpblock_8_depthwise_BN/gamma*
_output_shapes
: 
a
VarIsInitializedOp_135VarIsInitializedOpblock_16_depthwise_BN/gamma*
_output_shapes
: 
h
VarIsInitializedOp_136VarIsInitializedOp"block_7_project_BN/moving_variance*
_output_shapes
: 
f
VarIsInitializedOp_137VarIsInitializedOp block_4_depthwise_BN/moving_mean*
_output_shapes
: 
h
VarIsInitializedOp_138VarIsInitializedOp"block_6_depthwise/depthwise_kernel*
_output_shapes
: 
]
VarIsInitializedOp_139VarIsInitializedOpblock_11_expand_BN/beta*
_output_shapes
: 
e
VarIsInitializedOp_140VarIsInitializedOpexpanded_conv_depthwise_BN/beta*
_output_shapes
: 
]
VarIsInitializedOp_141VarIsInitializedOpblock_10_expand_BN/beta*
_output_shapes
: 
i
VarIsInitializedOp_142VarIsInitializedOp#block_15_depthwise/depthwise_kernel*
_output_shapes
: 
h
VarIsInitializedOp_143VarIsInitializedOp"block_2_depthwise/depthwise_kernel*
_output_shapes
: 
l
VarIsInitializedOp_144VarIsInitializedOp&expanded_conv_depthwise_BN/moving_mean*
_output_shapes
: 
[
VarIsInitializedOp_145VarIsInitializedOpblock_9_expand/kernel*
_output_shapes
: 
]
VarIsInitializedOp_146VarIsInitializedOpblock_11_project/kernel*
_output_shapes
: 
^
VarIsInitializedOp_147VarIsInitializedOpblock_3_project_BN/gamma*
_output_shapes
: 
e
VarIsInitializedOp_148VarIsInitializedOpblock_14_project_BN/moving_mean*
_output_shapes
: 
`
VarIsInitializedOp_149VarIsInitializedOpblock_2_depthwise_BN/gamma*
_output_shapes
: 
[
VarIsInitializedOp_150VarIsInitializedOpblock_4_expand/kernel*
_output_shapes
: 
^
VarIsInitializedOp_151VarIsInitializedOpblock_9_project_BN/gamma*
_output_shapes
: 
_
VarIsInitializedOp_152VarIsInitializedOpblock_8_depthwise_BN/beta*
_output_shapes
: 
^
VarIsInitializedOp_153VarIsInitializedOpblock_12_project_BN/beta*
_output_shapes
: 
i
VarIsInitializedOp_154VarIsInitializedOp#block_14_depthwise/depthwise_kernel*
_output_shapes
: 
f
VarIsInitializedOp_155VarIsInitializedOp block_8_depthwise_BN/moving_mean*
_output_shapes
: 
e
VarIsInitializedOp_156VarIsInitializedOpblock_12_project_BN/moving_mean*
_output_shapes
: 
i
VarIsInitializedOp_157VarIsInitializedOp#block_14_project_BN/moving_variance*
_output_shapes
: 
^
VarIsInitializedOp_158VarIsInitializedOpblock_4_project_BN/gamma*
_output_shapes
: 
`
VarIsInitializedOp_159VarIsInitializedOpblock_7_depthwise_BN/gamma*
_output_shapes
: 
\
VarIsInitializedOp_160VarIsInitializedOpblock_9_project/kernel*
_output_shapes
: 
a
VarIsInitializedOp_161VarIsInitializedOpblock_14_depthwise_BN/gamma*
_output_shapes
: 
_
VarIsInitializedOp_162VarIsInitializedOpblock_2_depthwise_BN/beta*
_output_shapes
: 
\
VarIsInitializedOp_163VarIsInitializedOpblock_16_expand/kernel*
_output_shapes
: 
]
VarIsInitializedOp_164VarIsInitializedOpblock_3_project_BN/beta*
_output_shapes
: 
j
VarIsInitializedOp_165VarIsInitializedOp$block_8_depthwise_BN/moving_variance*
_output_shapes
: 
i
VarIsInitializedOp_166VarIsInitializedOp#block_12_project_BN/moving_variance*
_output_shapes
: 
[
VarIsInitializedOp_167VarIsInitializedOpblock_8_expand/kernel*
_output_shapes
: 
f
VarIsInitializedOp_168VarIsInitializedOp block_2_depthwise_BN/moving_mean*
_output_shapes
: 
T
VarIsInitializedOp_169VarIsInitializedOpbn_Conv1/gamma*
_output_shapes
: 
_
VarIsInitializedOp_170VarIsInitializedOpblock_7_depthwise_BN/beta*
_output_shapes
: 
d
VarIsInitializedOp_171VarIsInitializedOpblock_3_project_BN/moving_mean*
_output_shapes
: 
]
VarIsInitializedOp_172VarIsInitializedOpblock_2_expand_BN/gamma*
_output_shapes
: 
a
VarIsInitializedOp_173VarIsInitializedOpblock_12_depthwise_BN/gamma*
_output_shapes
: 
]
VarIsInitializedOp_174VarIsInitializedOpblock_9_project_BN/beta*
_output_shapes
: 
i
VarIsInitializedOp_175VarIsInitializedOp#block_10_depthwise/depthwise_kernel*
_output_shapes
: 
f
VarIsInitializedOp_176VarIsInitializedOp block_7_depthwise_BN/moving_mean*
_output_shapes
: 
`
VarIsInitializedOp_177VarIsInitializedOpblock_14_depthwise_BN/beta*
_output_shapes
: 
\
VarIsInitializedOp_178VarIsInitializedOpblock_15_expand/kernel*
_output_shapes
: 
^
VarIsInitializedOp_179VarIsInitializedOpblock_15_expand_BN/gamma*
_output_shapes
: 
h
VarIsInitializedOp_180VarIsInitializedOp"block_5_depthwise/depthwise_kernel*
_output_shapes
: 
j
VarIsInitializedOp_181VarIsInitializedOp$block_2_depthwise_BN/moving_variance*
_output_shapes
: 
]
VarIsInitializedOp_182VarIsInitializedOpblock_4_project_BN/beta*
_output_shapes
: 
d
VarIsInitializedOp_183VarIsInitializedOpblock_9_project_BN/moving_mean*
_output_shapes
: 
[
VarIsInitializedOp_184VarIsInitializedOpblock_2_expand/kernel*
_output_shapes
: 
\
VarIsInitializedOp_185VarIsInitializedOpblock_8_project/kernel*
_output_shapes
: 
a
VarIsInitializedOp_186VarIsInitializedOpblock_13_depthwise_BN/gamma*
_output_shapes
: 
g
VarIsInitializedOp_187VarIsInitializedOp!block_14_depthwise_BN/moving_mean*
_output_shapes
: 
h
VarIsInitializedOp_188VarIsInitializedOp"block_3_project_BN/moving_variance*
_output_shapes
: 
i
VarIsInitializedOp_189VarIsInitializedOp#block_13_project_BN/moving_variance*
_output_shapes
: 
d
VarIsInitializedOp_190VarIsInitializedOpblock_4_project_BN/moving_mean*
_output_shapes
: 
]
VarIsInitializedOp_191VarIsInitializedOpblock_1_expand_BN/gamma*
_output_shapes
: 
`
VarIsInitializedOp_192VarIsInitializedOpblock_5_depthwise_BN/gamma*
_output_shapes
: 
S
VarIsInitializedOp_193VarIsInitializedOpConv_1/kernel*
_output_shapes
: 
^
VarIsInitializedOp_194VarIsInitializedOpblock_2_project_BN/gamma*
_output_shapes
: 
j
VarIsInitializedOp_195VarIsInitializedOp$block_5_depthwise_BN/moving_variance*
_output_shapes
: 
j
VarIsInitializedOp_196VarIsInitializedOp$block_7_depthwise_BN/moving_variance*
_output_shapes
: 
`
VarIsInitializedOp_197VarIsInitializedOpblock_12_depthwise_BN/beta*
_output_shapes
: 
\
VarIsInitializedOp_198VarIsInitializedOpblock_6_project/kernel*
_output_shapes
: 
^
VarIsInitializedOp_199VarIsInitializedOpblock_13_expand_BN/gamma*
_output_shapes
: 
_
VarIsInitializedOp_200VarIsInitializedOpblock_13_project_BN/gamma*
_output_shapes
: 
_
VarIsInitializedOp_201VarIsInitializedOpblock_16_project_BN/gamma*
_output_shapes
: 
h
VarIsInitializedOp_202VarIsInitializedOp"block_9_project_BN/moving_variance*
_output_shapes
: 
]
VarIsInitializedOp_203VarIsInitializedOpblock_15_expand_BN/beta*
_output_shapes
: 
`
VarIsInitializedOp_204VarIsInitializedOpblock_3_depthwise_BN/gamma*
_output_shapes
: 
g
VarIsInitializedOp_205VarIsInitializedOp!block_12_depthwise_BN/moving_mean*
_output_shapes
: 
k
VarIsInitializedOp_206VarIsInitializedOp%block_14_depthwise_BN/moving_variance*
_output_shapes
: 
j
VarIsInitializedOp_207VarIsInitializedOp$block_3_depthwise_BN/moving_variance*
_output_shapes
: 
i
VarIsInitializedOp_208VarIsInitializedOp#block_15_project_BN/moving_variance*
_output_shapes
: 
\
VarIsInitializedOp_209VarIsInitializedOpblock_2_expand_BN/beta*
_output_shapes
: 
h
VarIsInitializedOp_210VarIsInitializedOp"block_9_depthwise/depthwise_kernel*
_output_shapes
: 
h
VarIsInitializedOp_211VarIsInitializedOp"block_4_project_BN/moving_variance*
_output_shapes
: 
[
VarIsInitializedOp_212VarIsInitializedOpblock_6_expand/kernel*
_output_shapes
: 
d
VarIsInitializedOp_213VarIsInitializedOpblock_15_expand_BN/moving_mean*
_output_shapes
: 
_
VarIsInitializedOp_214VarIsInitializedOpblock_5_depthwise_BN/beta*
_output_shapes
: 
^
VarIsInitializedOp_215VarIsInitializedOpblock_12_expand_BN/gamma*
_output_shapes
: 
^
VarIsInitializedOp_216VarIsInitializedOpblock_16_project_BN/beta*
_output_shapes
: 
U
VarIsInitializedOp_217VarIsInitializedOpConv_1_bn/gamma*
_output_shapes
: 
`
VarIsInitializedOp_218VarIsInitializedOpblock_9_depthwise_BN/gamma*
_output_shapes
: 
k
VarIsInitializedOp_219VarIsInitializedOp%block_10_depthwise_BN/moving_variance*
_output_shapes
: 
]
VarIsInitializedOp_220VarIsInitializedOpblock_13_expand_BN/beta*
_output_shapes
: 
^
VarIsInitializedOp_221VarIsInitializedOpblock_13_project_BN/beta*
_output_shapes
: 
k
VarIsInitializedOp_222VarIsInitializedOp%block_12_depthwise_BN/moving_variance*
_output_shapes
: 
p
VarIsInitializedOp_223VarIsInitializedOp*expanded_conv_depthwise_BN/moving_variance*
_output_shapes
: 
f
VarIsInitializedOp_224VarIsInitializedOp block_5_depthwise_BN/moving_mean*
_output_shapes
: 
c
VarIsInitializedOp_225VarIsInitializedOpblock_2_expand_BN/moving_mean*
_output_shapes
: 
_
VarIsInitializedOp_226VarIsInitializedOpblock_3_depthwise_BN/beta*
_output_shapes
: 
e
VarIsInitializedOp_227VarIsInitializedOpblock_16_project_BN/moving_mean*
_output_shapes
: 
]
VarIsInitializedOp_228VarIsInitializedOpblock_5_expand_BN/gamma*
_output_shapes
: 
d
VarIsInitializedOp_229VarIsInitializedOpblock_13_expand_BN/moving_mean*
_output_shapes
: 
e
VarIsInitializedOp_230VarIsInitializedOpblock_13_project_BN/moving_mean*
_output_shapes
: 
h
VarIsInitializedOp_231VarIsInitializedOp"block_15_expand_BN/moving_variance*
_output_shapes
: 
\
VarIsInitializedOp_232VarIsInitializedOpblock_1_expand_BN/beta*
_output_shapes
: 
S
VarIsInitializedOp_233VarIsInitializedOpbn_Conv1/beta*
_output_shapes
: 
]
VarIsInitializedOp_234VarIsInitializedOpblock_2_project_BN/beta*
_output_shapes
: 
f
VarIsInitializedOp_235VarIsInitializedOp block_3_depthwise_BN/moving_mean*
_output_shapes
: 
_
VarIsInitializedOp_236VarIsInitializedOpblock_14_project_BN/gamma*
_output_shapes
: 
c
VarIsInitializedOp_237VarIsInitializedOpblock_1_expand_BN/moving_mean*
_output_shapes
: 
]
VarIsInitializedOp_238VarIsInitializedOpblock_12_expand_BN/beta*
_output_shapes
: 
d
VarIsInitializedOp_239VarIsInitializedOpblock_2_project_BN/moving_mean*
_output_shapes
: 
h
VarIsInitializedOp_240VarIsInitializedOp"block_8_depthwise/depthwise_kernel*
_output_shapes
: 
_
VarIsInitializedOp_241VarIsInitializedOpblock_9_depthwise_BN/beta*
_output_shapes
: 
i
VarIsInitializedOp_242VarIsInitializedOp#block_16_depthwise/depthwise_kernel*
_output_shapes
: 
g
VarIsInitializedOp_243VarIsInitializedOp!block_2_expand_BN/moving_variance*
_output_shapes
: 
^
VarIsInitializedOp_244VarIsInitializedOpbn_Conv1/moving_variance*
_output_shapes
: 
T
VarIsInitializedOp_245VarIsInitializedOpConv_1_bn/beta*
_output_shapes
: 
Z
VarIsInitializedOp_246VarIsInitializedOpbn_Conv1/moving_mean*
_output_shapes
: 
d
VarIsInitializedOp_247VarIsInitializedOpblock_12_expand_BN/moving_mean*
_output_shapes
: 
]
VarIsInitializedOp_248VarIsInitializedOpblock_12_project/kernel*
_output_shapes
: 
a
VarIsInitializedOp_249VarIsInitializedOpblock_15_depthwise_BN/gamma*
_output_shapes
: 
h
VarIsInitializedOp_250VarIsInitializedOp"block_1_depthwise/depthwise_kernel*
_output_shapes
: 
f
VarIsInitializedOp_251VarIsInitializedOp block_9_depthwise_BN/moving_mean*
_output_shapes
: 
h
VarIsInitializedOp_252VarIsInitializedOp"block_14_expand_BN/moving_variance*
_output_shapes
: 
_
VarIsInitializedOp_253VarIsInitializedOpblock_12_project_BN/gamma*
_output_shapes
: 
\
VarIsInitializedOp_254VarIsInitializedOpblock_5_expand_BN/beta*
_output_shapes
: 
^
VarIsInitializedOp_255VarIsInitializedOpblock_14_project_BN/beta*
_output_shapes
: 
[
VarIsInitializedOp_256VarIsInitializedOpConv_1_bn/moving_mean*
_output_shapes
: 
^
VarIsInitializedOp_257VarIsInitializedOpblock_10_project_BN/beta*
_output_shapes
: 
_
VarIsInitializedOp_258VarIsInitializedOpConv_1_bn/moving_variance*
_output_shapes
: 
\
VarIsInitializedOp_259VarIsInitializedOpblock_5_project/kernel*
_output_shapes
: 
�M
initNoOp^Conv1/kernel/Assign^Conv_1/kernel/Assign^Conv_1_bn/beta/Assign^Conv_1_bn/gamma/Assign^Conv_1_bn/moving_mean/Assign!^Conv_1_bn/moving_variance/Assign+^block_10_depthwise/depthwise_kernel/Assign"^block_10_depthwise_BN/beta/Assign#^block_10_depthwise_BN/gamma/Assign)^block_10_depthwise_BN/moving_mean/Assign-^block_10_depthwise_BN/moving_variance/Assign^block_10_expand/kernel/Assign^block_10_expand_BN/beta/Assign ^block_10_expand_BN/gamma/Assign&^block_10_expand_BN/moving_mean/Assign*^block_10_expand_BN/moving_variance/Assign^block_10_project/kernel/Assign ^block_10_project_BN/beta/Assign!^block_10_project_BN/gamma/Assign'^block_10_project_BN/moving_mean/Assign+^block_10_project_BN/moving_variance/Assign+^block_11_depthwise/depthwise_kernel/Assign"^block_11_depthwise_BN/beta/Assign#^block_11_depthwise_BN/gamma/Assign)^block_11_depthwise_BN/moving_mean/Assign-^block_11_depthwise_BN/moving_variance/Assign^block_11_expand/kernel/Assign^block_11_expand_BN/beta/Assign ^block_11_expand_BN/gamma/Assign&^block_11_expand_BN/moving_mean/Assign*^block_11_expand_BN/moving_variance/Assign^block_11_project/kernel/Assign ^block_11_project_BN/beta/Assign!^block_11_project_BN/gamma/Assign'^block_11_project_BN/moving_mean/Assign+^block_11_project_BN/moving_variance/Assign+^block_12_depthwise/depthwise_kernel/Assign"^block_12_depthwise_BN/beta/Assign#^block_12_depthwise_BN/gamma/Assign)^block_12_depthwise_BN/moving_mean/Assign-^block_12_depthwise_BN/moving_variance/Assign^block_12_expand/kernel/Assign^block_12_expand_BN/beta/Assign ^block_12_expand_BN/gamma/Assign&^block_12_expand_BN/moving_mean/Assign*^block_12_expand_BN/moving_variance/Assign^block_12_project/kernel/Assign ^block_12_project_BN/beta/Assign!^block_12_project_BN/gamma/Assign'^block_12_project_BN/moving_mean/Assign+^block_12_project_BN/moving_variance/Assign+^block_13_depthwise/depthwise_kernel/Assign"^block_13_depthwise_BN/beta/Assign#^block_13_depthwise_BN/gamma/Assign)^block_13_depthwise_BN/moving_mean/Assign-^block_13_depthwise_BN/moving_variance/Assign^block_13_expand/kernel/Assign^block_13_expand_BN/beta/Assign ^block_13_expand_BN/gamma/Assign&^block_13_expand_BN/moving_mean/Assign*^block_13_expand_BN/moving_variance/Assign^block_13_project/kernel/Assign ^block_13_project_BN/beta/Assign!^block_13_project_BN/gamma/Assign'^block_13_project_BN/moving_mean/Assign+^block_13_project_BN/moving_variance/Assign+^block_14_depthwise/depthwise_kernel/Assign"^block_14_depthwise_BN/beta/Assign#^block_14_depthwise_BN/gamma/Assign)^block_14_depthwise_BN/moving_mean/Assign-^block_14_depthwise_BN/moving_variance/Assign^block_14_expand/kernel/Assign^block_14_expand_BN/beta/Assign ^block_14_expand_BN/gamma/Assign&^block_14_expand_BN/moving_mean/Assign*^block_14_expand_BN/moving_variance/Assign^block_14_project/kernel/Assign ^block_14_project_BN/beta/Assign!^block_14_project_BN/gamma/Assign'^block_14_project_BN/moving_mean/Assign+^block_14_project_BN/moving_variance/Assign+^block_15_depthwise/depthwise_kernel/Assign"^block_15_depthwise_BN/beta/Assign#^block_15_depthwise_BN/gamma/Assign)^block_15_depthwise_BN/moving_mean/Assign-^block_15_depthwise_BN/moving_variance/Assign^block_15_expand/kernel/Assign^block_15_expand_BN/beta/Assign ^block_15_expand_BN/gamma/Assign&^block_15_expand_BN/moving_mean/Assign*^block_15_expand_BN/moving_variance/Assign^block_15_project/kernel/Assign ^block_15_project_BN/beta/Assign!^block_15_project_BN/gamma/Assign'^block_15_project_BN/moving_mean/Assign+^block_15_project_BN/moving_variance/Assign+^block_16_depthwise/depthwise_kernel/Assign"^block_16_depthwise_BN/beta/Assign#^block_16_depthwise_BN/gamma/Assign)^block_16_depthwise_BN/moving_mean/Assign-^block_16_depthwise_BN/moving_variance/Assign^block_16_expand/kernel/Assign^block_16_expand_BN/beta/Assign ^block_16_expand_BN/gamma/Assign&^block_16_expand_BN/moving_mean/Assign*^block_16_expand_BN/moving_variance/Assign^block_16_project/kernel/Assign ^block_16_project_BN/beta/Assign!^block_16_project_BN/gamma/Assign'^block_16_project_BN/moving_mean/Assign+^block_16_project_BN/moving_variance/Assign*^block_1_depthwise/depthwise_kernel/Assign!^block_1_depthwise_BN/beta/Assign"^block_1_depthwise_BN/gamma/Assign(^block_1_depthwise_BN/moving_mean/Assign,^block_1_depthwise_BN/moving_variance/Assign^block_1_expand/kernel/Assign^block_1_expand_BN/beta/Assign^block_1_expand_BN/gamma/Assign%^block_1_expand_BN/moving_mean/Assign)^block_1_expand_BN/moving_variance/Assign^block_1_project/kernel/Assign^block_1_project_BN/beta/Assign ^block_1_project_BN/gamma/Assign&^block_1_project_BN/moving_mean/Assign*^block_1_project_BN/moving_variance/Assign*^block_2_depthwise/depthwise_kernel/Assign!^block_2_depthwise_BN/beta/Assign"^block_2_depthwise_BN/gamma/Assign(^block_2_depthwise_BN/moving_mean/Assign,^block_2_depthwise_BN/moving_variance/Assign^block_2_expand/kernel/Assign^block_2_expand_BN/beta/Assign^block_2_expand_BN/gamma/Assign%^block_2_expand_BN/moving_mean/Assign)^block_2_expand_BN/moving_variance/Assign^block_2_project/kernel/Assign^block_2_project_BN/beta/Assign ^block_2_project_BN/gamma/Assign&^block_2_project_BN/moving_mean/Assign*^block_2_project_BN/moving_variance/Assign*^block_3_depthwise/depthwise_kernel/Assign!^block_3_depthwise_BN/beta/Assign"^block_3_depthwise_BN/gamma/Assign(^block_3_depthwise_BN/moving_mean/Assign,^block_3_depthwise_BN/moving_variance/Assign^block_3_expand/kernel/Assign^block_3_expand_BN/beta/Assign^block_3_expand_BN/gamma/Assign%^block_3_expand_BN/moving_mean/Assign)^block_3_expand_BN/moving_variance/Assign^block_3_project/kernel/Assign^block_3_project_BN/beta/Assign ^block_3_project_BN/gamma/Assign&^block_3_project_BN/moving_mean/Assign*^block_3_project_BN/moving_variance/Assign*^block_4_depthwise/depthwise_kernel/Assign!^block_4_depthwise_BN/beta/Assign"^block_4_depthwise_BN/gamma/Assign(^block_4_depthwise_BN/moving_mean/Assign,^block_4_depthwise_BN/moving_variance/Assign^block_4_expand/kernel/Assign^block_4_expand_BN/beta/Assign^block_4_expand_BN/gamma/Assign%^block_4_expand_BN/moving_mean/Assign)^block_4_expand_BN/moving_variance/Assign^block_4_project/kernel/Assign^block_4_project_BN/beta/Assign ^block_4_project_BN/gamma/Assign&^block_4_project_BN/moving_mean/Assign*^block_4_project_BN/moving_variance/Assign*^block_5_depthwise/depthwise_kernel/Assign!^block_5_depthwise_BN/beta/Assign"^block_5_depthwise_BN/gamma/Assign(^block_5_depthwise_BN/moving_mean/Assign,^block_5_depthwise_BN/moving_variance/Assign^block_5_expand/kernel/Assign^block_5_expand_BN/beta/Assign^block_5_expand_BN/gamma/Assign%^block_5_expand_BN/moving_mean/Assign)^block_5_expand_BN/moving_variance/Assign^block_5_project/kernel/Assign^block_5_project_BN/beta/Assign ^block_5_project_BN/gamma/Assign&^block_5_project_BN/moving_mean/Assign*^block_5_project_BN/moving_variance/Assign*^block_6_depthwise/depthwise_kernel/Assign!^block_6_depthwise_BN/beta/Assign"^block_6_depthwise_BN/gamma/Assign(^block_6_depthwise_BN/moving_mean/Assign,^block_6_depthwise_BN/moving_variance/Assign^block_6_expand/kernel/Assign^block_6_expand_BN/beta/Assign^block_6_expand_BN/gamma/Assign%^block_6_expand_BN/moving_mean/Assign)^block_6_expand_BN/moving_variance/Assign^block_6_project/kernel/Assign^block_6_project_BN/beta/Assign ^block_6_project_BN/gamma/Assign&^block_6_project_BN/moving_mean/Assign*^block_6_project_BN/moving_variance/Assign*^block_7_depthwise/depthwise_kernel/Assign!^block_7_depthwise_BN/beta/Assign"^block_7_depthwise_BN/gamma/Assign(^block_7_depthwise_BN/moving_mean/Assign,^block_7_depthwise_BN/moving_variance/Assign^block_7_expand/kernel/Assign^block_7_expand_BN/beta/Assign^block_7_expand_BN/gamma/Assign%^block_7_expand_BN/moving_mean/Assign)^block_7_expand_BN/moving_variance/Assign^block_7_project/kernel/Assign^block_7_project_BN/beta/Assign ^block_7_project_BN/gamma/Assign&^block_7_project_BN/moving_mean/Assign*^block_7_project_BN/moving_variance/Assign*^block_8_depthwise/depthwise_kernel/Assign!^block_8_depthwise_BN/beta/Assign"^block_8_depthwise_BN/gamma/Assign(^block_8_depthwise_BN/moving_mean/Assign,^block_8_depthwise_BN/moving_variance/Assign^block_8_expand/kernel/Assign^block_8_expand_BN/beta/Assign^block_8_expand_BN/gamma/Assign%^block_8_expand_BN/moving_mean/Assign)^block_8_expand_BN/moving_variance/Assign^block_8_project/kernel/Assign^block_8_project_BN/beta/Assign ^block_8_project_BN/gamma/Assign&^block_8_project_BN/moving_mean/Assign*^block_8_project_BN/moving_variance/Assign*^block_9_depthwise/depthwise_kernel/Assign!^block_9_depthwise_BN/beta/Assign"^block_9_depthwise_BN/gamma/Assign(^block_9_depthwise_BN/moving_mean/Assign,^block_9_depthwise_BN/moving_variance/Assign^block_9_expand/kernel/Assign^block_9_expand_BN/beta/Assign^block_9_expand_BN/gamma/Assign%^block_9_expand_BN/moving_mean/Assign)^block_9_expand_BN/moving_variance/Assign^block_9_project/kernel/Assign^block_9_project_BN/beta/Assign ^block_9_project_BN/gamma/Assign&^block_9_project_BN/moving_mean/Assign*^block_9_project_BN/moving_variance/Assign^bn_Conv1/beta/Assign^bn_Conv1/gamma/Assign^bn_Conv1/moving_mean/Assign ^bn_Conv1/moving_variance/Assign0^expanded_conv_depthwise/depthwise_kernel/Assign'^expanded_conv_depthwise_BN/beta/Assign(^expanded_conv_depthwise_BN/gamma/Assign.^expanded_conv_depthwise_BN/moving_mean/Assign2^expanded_conv_depthwise_BN/moving_variance/Assign$^expanded_conv_project/kernel/Assign%^expanded_conv_project_BN/beta/Assign&^expanded_conv_project_BN/gamma/Assign,^expanded_conv_project_BN/moving_mean/Assign0^expanded_conv_project_BN/moving_variance/Assign
W
Const_1Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_2Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_6e1500be56ba4aff9c5dd29e338c2a34/part*
dtype0*
_output_shapes
: 
f

StringJoin
StringJoinConst_2StringJoin/inputs_1"/device:CPU:0*
N*
_output_shapes
: 
L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
�z
SaveV2/tensor_namesConst"/device:CPU:0*�y
value�yB�y�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-20/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-23/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-23/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-23/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-25/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-25/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-25/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-26/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-27/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-27/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-27/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-29/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-29/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-29/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-31/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-31/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-31/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-31/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-32/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-33/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-33/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-33/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-33/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-35/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-35/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-35/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-35/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-37/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-37/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-37/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-37/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-38/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-39/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-39/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-39/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-39/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-41/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-41/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-41/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-41/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-42/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-43/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-43/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-43/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-43/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-44/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-45/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-45/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-45/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-45/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-46/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-47/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-47/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-47/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-47/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-48/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-49/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-49/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-49/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-49/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-50/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-51/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-51/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-51/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-51/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-52/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-53/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-53/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-53/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-53/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-54/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-55/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-55/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-55/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-55/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-56/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-57/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-57/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-57/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-57/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-58/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-59/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-59/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-59/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-59/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-60/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-61/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-61/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-61/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-61/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-62/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-63/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-63/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-63/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-63/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-64/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-65/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-65/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-65/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-65/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-66/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-67/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-67/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-67/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-67/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-68/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-69/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-69/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-69/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-69/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-70/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-71/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-71/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-71/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-71/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-72/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-73/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-73/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-73/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-73/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-74/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-75/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-75/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-75/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-75/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-76/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-77/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-77/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-77/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-77/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-78/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-79/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-79/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-79/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-79/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-80/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-81/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-81/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-81/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-81/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-82/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-83/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-83/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-83/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-83/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-84/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-85/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-85/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-85/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-85/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-86/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-87/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-87/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-87/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-87/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-88/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-89/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-89/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-89/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-89/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-90/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-91/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-91/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-91/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-91/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-92/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-93/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-93/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-93/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-93/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-94/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-95/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-95/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-95/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-95/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-96/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-97/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-97/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-97/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-97/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-98/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-99/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-99/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-99/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-99/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-100/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-101/gamma/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-101/beta/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-101/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-101/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-102/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-103/gamma/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-103/beta/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-103/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-103/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes	
:�
�
SaveV2/shape_and_slicesConst"/device:CPU:0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:�
�h
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices Conv1/kernel/Read/ReadVariableOp"bn_Conv1/gamma/Read/ReadVariableOp!bn_Conv1/beta/Read/ReadVariableOp(bn_Conv1/moving_mean/Read/ReadVariableOp,bn_Conv1/moving_variance/Read/ReadVariableOp<expanded_conv_depthwise/depthwise_kernel/Read/ReadVariableOp4expanded_conv_depthwise_BN/gamma/Read/ReadVariableOp3expanded_conv_depthwise_BN/beta/Read/ReadVariableOp:expanded_conv_depthwise_BN/moving_mean/Read/ReadVariableOp>expanded_conv_depthwise_BN/moving_variance/Read/ReadVariableOp0expanded_conv_project/kernel/Read/ReadVariableOp2expanded_conv_project_BN/gamma/Read/ReadVariableOp1expanded_conv_project_BN/beta/Read/ReadVariableOp8expanded_conv_project_BN/moving_mean/Read/ReadVariableOp<expanded_conv_project_BN/moving_variance/Read/ReadVariableOp)block_1_expand/kernel/Read/ReadVariableOp+block_1_expand_BN/gamma/Read/ReadVariableOp*block_1_expand_BN/beta/Read/ReadVariableOp1block_1_expand_BN/moving_mean/Read/ReadVariableOp5block_1_expand_BN/moving_variance/Read/ReadVariableOp6block_1_depthwise/depthwise_kernel/Read/ReadVariableOp.block_1_depthwise_BN/gamma/Read/ReadVariableOp-block_1_depthwise_BN/beta/Read/ReadVariableOp4block_1_depthwise_BN/moving_mean/Read/ReadVariableOp8block_1_depthwise_BN/moving_variance/Read/ReadVariableOp*block_1_project/kernel/Read/ReadVariableOp,block_1_project_BN/gamma/Read/ReadVariableOp+block_1_project_BN/beta/Read/ReadVariableOp2block_1_project_BN/moving_mean/Read/ReadVariableOp6block_1_project_BN/moving_variance/Read/ReadVariableOp)block_2_expand/kernel/Read/ReadVariableOp+block_2_expand_BN/gamma/Read/ReadVariableOp*block_2_expand_BN/beta/Read/ReadVariableOp1block_2_expand_BN/moving_mean/Read/ReadVariableOp5block_2_expand_BN/moving_variance/Read/ReadVariableOp6block_2_depthwise/depthwise_kernel/Read/ReadVariableOp.block_2_depthwise_BN/gamma/Read/ReadVariableOp-block_2_depthwise_BN/beta/Read/ReadVariableOp4block_2_depthwise_BN/moving_mean/Read/ReadVariableOp8block_2_depthwise_BN/moving_variance/Read/ReadVariableOp*block_2_project/kernel/Read/ReadVariableOp,block_2_project_BN/gamma/Read/ReadVariableOp+block_2_project_BN/beta/Read/ReadVariableOp2block_2_project_BN/moving_mean/Read/ReadVariableOp6block_2_project_BN/moving_variance/Read/ReadVariableOp)block_3_expand/kernel/Read/ReadVariableOp+block_3_expand_BN/gamma/Read/ReadVariableOp*block_3_expand_BN/beta/Read/ReadVariableOp1block_3_expand_BN/moving_mean/Read/ReadVariableOp5block_3_expand_BN/moving_variance/Read/ReadVariableOp6block_3_depthwise/depthwise_kernel/Read/ReadVariableOp.block_3_depthwise_BN/gamma/Read/ReadVariableOp-block_3_depthwise_BN/beta/Read/ReadVariableOp4block_3_depthwise_BN/moving_mean/Read/ReadVariableOp8block_3_depthwise_BN/moving_variance/Read/ReadVariableOp*block_3_project/kernel/Read/ReadVariableOp,block_3_project_BN/gamma/Read/ReadVariableOp+block_3_project_BN/beta/Read/ReadVariableOp2block_3_project_BN/moving_mean/Read/ReadVariableOp6block_3_project_BN/moving_variance/Read/ReadVariableOp)block_4_expand/kernel/Read/ReadVariableOp+block_4_expand_BN/gamma/Read/ReadVariableOp*block_4_expand_BN/beta/Read/ReadVariableOp1block_4_expand_BN/moving_mean/Read/ReadVariableOp5block_4_expand_BN/moving_variance/Read/ReadVariableOp6block_4_depthwise/depthwise_kernel/Read/ReadVariableOp.block_4_depthwise_BN/gamma/Read/ReadVariableOp-block_4_depthwise_BN/beta/Read/ReadVariableOp4block_4_depthwise_BN/moving_mean/Read/ReadVariableOp8block_4_depthwise_BN/moving_variance/Read/ReadVariableOp*block_4_project/kernel/Read/ReadVariableOp,block_4_project_BN/gamma/Read/ReadVariableOp+block_4_project_BN/beta/Read/ReadVariableOp2block_4_project_BN/moving_mean/Read/ReadVariableOp6block_4_project_BN/moving_variance/Read/ReadVariableOp)block_5_expand/kernel/Read/ReadVariableOp+block_5_expand_BN/gamma/Read/ReadVariableOp*block_5_expand_BN/beta/Read/ReadVariableOp1block_5_expand_BN/moving_mean/Read/ReadVariableOp5block_5_expand_BN/moving_variance/Read/ReadVariableOp6block_5_depthwise/depthwise_kernel/Read/ReadVariableOp.block_5_depthwise_BN/gamma/Read/ReadVariableOp-block_5_depthwise_BN/beta/Read/ReadVariableOp4block_5_depthwise_BN/moving_mean/Read/ReadVariableOp8block_5_depthwise_BN/moving_variance/Read/ReadVariableOp*block_5_project/kernel/Read/ReadVariableOp,block_5_project_BN/gamma/Read/ReadVariableOp+block_5_project_BN/beta/Read/ReadVariableOp2block_5_project_BN/moving_mean/Read/ReadVariableOp6block_5_project_BN/moving_variance/Read/ReadVariableOp)block_6_expand/kernel/Read/ReadVariableOp+block_6_expand_BN/gamma/Read/ReadVariableOp*block_6_expand_BN/beta/Read/ReadVariableOp1block_6_expand_BN/moving_mean/Read/ReadVariableOp5block_6_expand_BN/moving_variance/Read/ReadVariableOp6block_6_depthwise/depthwise_kernel/Read/ReadVariableOp.block_6_depthwise_BN/gamma/Read/ReadVariableOp-block_6_depthwise_BN/beta/Read/ReadVariableOp4block_6_depthwise_BN/moving_mean/Read/ReadVariableOp8block_6_depthwise_BN/moving_variance/Read/ReadVariableOp*block_6_project/kernel/Read/ReadVariableOp,block_6_project_BN/gamma/Read/ReadVariableOp+block_6_project_BN/beta/Read/ReadVariableOp2block_6_project_BN/moving_mean/Read/ReadVariableOp6block_6_project_BN/moving_variance/Read/ReadVariableOp)block_7_expand/kernel/Read/ReadVariableOp+block_7_expand_BN/gamma/Read/ReadVariableOp*block_7_expand_BN/beta/Read/ReadVariableOp1block_7_expand_BN/moving_mean/Read/ReadVariableOp5block_7_expand_BN/moving_variance/Read/ReadVariableOp6block_7_depthwise/depthwise_kernel/Read/ReadVariableOp.block_7_depthwise_BN/gamma/Read/ReadVariableOp-block_7_depthwise_BN/beta/Read/ReadVariableOp4block_7_depthwise_BN/moving_mean/Read/ReadVariableOp8block_7_depthwise_BN/moving_variance/Read/ReadVariableOp*block_7_project/kernel/Read/ReadVariableOp,block_7_project_BN/gamma/Read/ReadVariableOp+block_7_project_BN/beta/Read/ReadVariableOp2block_7_project_BN/moving_mean/Read/ReadVariableOp6block_7_project_BN/moving_variance/Read/ReadVariableOp)block_8_expand/kernel/Read/ReadVariableOp+block_8_expand_BN/gamma/Read/ReadVariableOp*block_8_expand_BN/beta/Read/ReadVariableOp1block_8_expand_BN/moving_mean/Read/ReadVariableOp5block_8_expand_BN/moving_variance/Read/ReadVariableOp6block_8_depthwise/depthwise_kernel/Read/ReadVariableOp.block_8_depthwise_BN/gamma/Read/ReadVariableOp-block_8_depthwise_BN/beta/Read/ReadVariableOp4block_8_depthwise_BN/moving_mean/Read/ReadVariableOp8block_8_depthwise_BN/moving_variance/Read/ReadVariableOp*block_8_project/kernel/Read/ReadVariableOp,block_8_project_BN/gamma/Read/ReadVariableOp+block_8_project_BN/beta/Read/ReadVariableOp2block_8_project_BN/moving_mean/Read/ReadVariableOp6block_8_project_BN/moving_variance/Read/ReadVariableOp)block_9_expand/kernel/Read/ReadVariableOp+block_9_expand_BN/gamma/Read/ReadVariableOp*block_9_expand_BN/beta/Read/ReadVariableOp1block_9_expand_BN/moving_mean/Read/ReadVariableOp5block_9_expand_BN/moving_variance/Read/ReadVariableOp6block_9_depthwise/depthwise_kernel/Read/ReadVariableOp.block_9_depthwise_BN/gamma/Read/ReadVariableOp-block_9_depthwise_BN/beta/Read/ReadVariableOp4block_9_depthwise_BN/moving_mean/Read/ReadVariableOp8block_9_depthwise_BN/moving_variance/Read/ReadVariableOp*block_9_project/kernel/Read/ReadVariableOp,block_9_project_BN/gamma/Read/ReadVariableOp+block_9_project_BN/beta/Read/ReadVariableOp2block_9_project_BN/moving_mean/Read/ReadVariableOp6block_9_project_BN/moving_variance/Read/ReadVariableOp*block_10_expand/kernel/Read/ReadVariableOp,block_10_expand_BN/gamma/Read/ReadVariableOp+block_10_expand_BN/beta/Read/ReadVariableOp2block_10_expand_BN/moving_mean/Read/ReadVariableOp6block_10_expand_BN/moving_variance/Read/ReadVariableOp7block_10_depthwise/depthwise_kernel/Read/ReadVariableOp/block_10_depthwise_BN/gamma/Read/ReadVariableOp.block_10_depthwise_BN/beta/Read/ReadVariableOp5block_10_depthwise_BN/moving_mean/Read/ReadVariableOp9block_10_depthwise_BN/moving_variance/Read/ReadVariableOp+block_10_project/kernel/Read/ReadVariableOp-block_10_project_BN/gamma/Read/ReadVariableOp,block_10_project_BN/beta/Read/ReadVariableOp3block_10_project_BN/moving_mean/Read/ReadVariableOp7block_10_project_BN/moving_variance/Read/ReadVariableOp*block_11_expand/kernel/Read/ReadVariableOp,block_11_expand_BN/gamma/Read/ReadVariableOp+block_11_expand_BN/beta/Read/ReadVariableOp2block_11_expand_BN/moving_mean/Read/ReadVariableOp6block_11_expand_BN/moving_variance/Read/ReadVariableOp7block_11_depthwise/depthwise_kernel/Read/ReadVariableOp/block_11_depthwise_BN/gamma/Read/ReadVariableOp.block_11_depthwise_BN/beta/Read/ReadVariableOp5block_11_depthwise_BN/moving_mean/Read/ReadVariableOp9block_11_depthwise_BN/moving_variance/Read/ReadVariableOp+block_11_project/kernel/Read/ReadVariableOp-block_11_project_BN/gamma/Read/ReadVariableOp,block_11_project_BN/beta/Read/ReadVariableOp3block_11_project_BN/moving_mean/Read/ReadVariableOp7block_11_project_BN/moving_variance/Read/ReadVariableOp*block_12_expand/kernel/Read/ReadVariableOp,block_12_expand_BN/gamma/Read/ReadVariableOp+block_12_expand_BN/beta/Read/ReadVariableOp2block_12_expand_BN/moving_mean/Read/ReadVariableOp6block_12_expand_BN/moving_variance/Read/ReadVariableOp7block_12_depthwise/depthwise_kernel/Read/ReadVariableOp/block_12_depthwise_BN/gamma/Read/ReadVariableOp.block_12_depthwise_BN/beta/Read/ReadVariableOp5block_12_depthwise_BN/moving_mean/Read/ReadVariableOp9block_12_depthwise_BN/moving_variance/Read/ReadVariableOp+block_12_project/kernel/Read/ReadVariableOp-block_12_project_BN/gamma/Read/ReadVariableOp,block_12_project_BN/beta/Read/ReadVariableOp3block_12_project_BN/moving_mean/Read/ReadVariableOp7block_12_project_BN/moving_variance/Read/ReadVariableOp*block_13_expand/kernel/Read/ReadVariableOp,block_13_expand_BN/gamma/Read/ReadVariableOp+block_13_expand_BN/beta/Read/ReadVariableOp2block_13_expand_BN/moving_mean/Read/ReadVariableOp6block_13_expand_BN/moving_variance/Read/ReadVariableOp7block_13_depthwise/depthwise_kernel/Read/ReadVariableOp/block_13_depthwise_BN/gamma/Read/ReadVariableOp.block_13_depthwise_BN/beta/Read/ReadVariableOp5block_13_depthwise_BN/moving_mean/Read/ReadVariableOp9block_13_depthwise_BN/moving_variance/Read/ReadVariableOp+block_13_project/kernel/Read/ReadVariableOp-block_13_project_BN/gamma/Read/ReadVariableOp,block_13_project_BN/beta/Read/ReadVariableOp3block_13_project_BN/moving_mean/Read/ReadVariableOp7block_13_project_BN/moving_variance/Read/ReadVariableOp*block_14_expand/kernel/Read/ReadVariableOp,block_14_expand_BN/gamma/Read/ReadVariableOp+block_14_expand_BN/beta/Read/ReadVariableOp2block_14_expand_BN/moving_mean/Read/ReadVariableOp6block_14_expand_BN/moving_variance/Read/ReadVariableOp7block_14_depthwise/depthwise_kernel/Read/ReadVariableOp/block_14_depthwise_BN/gamma/Read/ReadVariableOp.block_14_depthwise_BN/beta/Read/ReadVariableOp5block_14_depthwise_BN/moving_mean/Read/ReadVariableOp9block_14_depthwise_BN/moving_variance/Read/ReadVariableOp+block_14_project/kernel/Read/ReadVariableOp-block_14_project_BN/gamma/Read/ReadVariableOp,block_14_project_BN/beta/Read/ReadVariableOp3block_14_project_BN/moving_mean/Read/ReadVariableOp7block_14_project_BN/moving_variance/Read/ReadVariableOp*block_15_expand/kernel/Read/ReadVariableOp,block_15_expand_BN/gamma/Read/ReadVariableOp+block_15_expand_BN/beta/Read/ReadVariableOp2block_15_expand_BN/moving_mean/Read/ReadVariableOp6block_15_expand_BN/moving_variance/Read/ReadVariableOp7block_15_depthwise/depthwise_kernel/Read/ReadVariableOp/block_15_depthwise_BN/gamma/Read/ReadVariableOp.block_15_depthwise_BN/beta/Read/ReadVariableOp5block_15_depthwise_BN/moving_mean/Read/ReadVariableOp9block_15_depthwise_BN/moving_variance/Read/ReadVariableOp+block_15_project/kernel/Read/ReadVariableOp-block_15_project_BN/gamma/Read/ReadVariableOp,block_15_project_BN/beta/Read/ReadVariableOp3block_15_project_BN/moving_mean/Read/ReadVariableOp7block_15_project_BN/moving_variance/Read/ReadVariableOp*block_16_expand/kernel/Read/ReadVariableOp,block_16_expand_BN/gamma/Read/ReadVariableOp+block_16_expand_BN/beta/Read/ReadVariableOp2block_16_expand_BN/moving_mean/Read/ReadVariableOp6block_16_expand_BN/moving_variance/Read/ReadVariableOp7block_16_depthwise/depthwise_kernel/Read/ReadVariableOp/block_16_depthwise_BN/gamma/Read/ReadVariableOp.block_16_depthwise_BN/beta/Read/ReadVariableOp5block_16_depthwise_BN/moving_mean/Read/ReadVariableOp9block_16_depthwise_BN/moving_variance/Read/ReadVariableOp+block_16_project/kernel/Read/ReadVariableOp-block_16_project_BN/gamma/Read/ReadVariableOp,block_16_project_BN/beta/Read/ReadVariableOp3block_16_project_BN/moving_mean/Read/ReadVariableOp7block_16_project_BN/moving_variance/Read/ReadVariableOp!Conv_1/kernel/Read/ReadVariableOp#Conv_1_bn/gamma/Read/ReadVariableOp"Conv_1_bn/beta/Read/ReadVariableOp)Conv_1_bn/moving_mean/Read/ReadVariableOp-Conv_1_bn/moving_variance/Read/ReadVariableOp"/device:CPU:0*�
dtypes�
�2�
h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
|
ShardedFilename_1ShardedFilename
StringJoinShardedFilename_1/shard
num_shards"/device:CPU:0*
_output_shapes
: 
�
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:
q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
SaveV2_1SaveV2ShardedFilename_1SaveV2_1/tensor_namesSaveV2_1/shape_and_slicesConst_1"/device:CPU:0*
dtypes
2
�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilenameShardedFilename_1^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:
h
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixesConst_2"/device:CPU:0
f
Identity_260IdentityConst_2^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
�z
save/SaveV2/tensor_namesConst*�y
value�yB�y�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-100/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-101/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-101/gamma/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-101/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-101/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-102/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-103/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-103/gamma/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-103/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-103/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-20/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-23/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-23/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-23/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-25/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-25/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-25/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-26/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-27/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-27/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-27/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-29/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-29/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-29/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-31/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-31/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-31/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-31/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-32/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-33/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-33/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-33/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-33/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-35/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-35/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-35/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-35/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-37/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-37/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-37/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-37/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-38/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-39/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-39/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-39/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-39/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-41/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-41/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-41/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-41/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-42/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-43/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-43/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-43/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-43/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-44/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-45/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-45/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-45/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-45/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-46/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-47/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-47/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-47/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-47/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-48/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-49/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-49/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-49/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-49/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-50/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-51/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-51/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-51/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-51/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-52/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-53/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-53/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-53/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-53/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-54/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-55/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-55/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-55/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-55/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-56/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-57/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-57/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-57/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-57/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-58/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-59/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-59/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-59/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-59/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-60/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-61/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-61/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-61/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-61/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-62/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-63/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-63/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-63/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-63/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-64/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-65/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-65/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-65/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-65/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-66/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-67/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-67/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-67/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-67/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-68/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-69/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-69/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-69/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-69/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-70/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-71/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-71/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-71/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-71/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-72/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-73/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-73/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-73/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-73/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-74/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-75/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-75/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-75/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-75/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-76/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-77/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-77/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-77/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-77/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-78/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-79/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-79/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-79/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-79/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-80/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-81/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-81/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-81/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-81/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-82/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-83/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-83/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-83/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-83/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-84/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-85/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-85/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-85/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-85/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-86/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-87/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-87/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-87/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-87/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-88/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-89/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-89/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-89/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-89/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-90/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-91/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-91/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-91/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-91/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-92/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-93/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-93/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-93/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-93/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-94/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-95/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-95/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-95/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-95/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-96/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-97/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-97/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-97/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-97/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-98/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-99/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-99/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-99/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-99/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes	
:�
�
save/SaveV2/shape_and_slicesConst*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:�
�h
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices Conv1/kernel/Read/ReadVariableOp!bn_Conv1/beta/Read/ReadVariableOp"bn_Conv1/gamma/Read/ReadVariableOp(bn_Conv1/moving_mean/Read/ReadVariableOp,bn_Conv1/moving_variance/Read/ReadVariableOp*block_1_project/kernel/Read/ReadVariableOp+block_16_project/kernel/Read/ReadVariableOp,block_16_project_BN/beta/Read/ReadVariableOp-block_16_project_BN/gamma/Read/ReadVariableOp3block_16_project_BN/moving_mean/Read/ReadVariableOp7block_16_project_BN/moving_variance/Read/ReadVariableOp!Conv_1/kernel/Read/ReadVariableOp"Conv_1_bn/beta/Read/ReadVariableOp#Conv_1_bn/gamma/Read/ReadVariableOp)Conv_1_bn/moving_mean/Read/ReadVariableOp-Conv_1_bn/moving_variance/Read/ReadVariableOp+block_1_project_BN/beta/Read/ReadVariableOp,block_1_project_BN/gamma/Read/ReadVariableOp2block_1_project_BN/moving_mean/Read/ReadVariableOp6block_1_project_BN/moving_variance/Read/ReadVariableOp)block_2_expand/kernel/Read/ReadVariableOp*block_2_expand_BN/beta/Read/ReadVariableOp+block_2_expand_BN/gamma/Read/ReadVariableOp1block_2_expand_BN/moving_mean/Read/ReadVariableOp5block_2_expand_BN/moving_variance/Read/ReadVariableOp6block_2_depthwise/depthwise_kernel/Read/ReadVariableOp-block_2_depthwise_BN/beta/Read/ReadVariableOp.block_2_depthwise_BN/gamma/Read/ReadVariableOp4block_2_depthwise_BN/moving_mean/Read/ReadVariableOp8block_2_depthwise_BN/moving_variance/Read/ReadVariableOp*block_2_project/kernel/Read/ReadVariableOp+block_2_project_BN/beta/Read/ReadVariableOp,block_2_project_BN/gamma/Read/ReadVariableOp2block_2_project_BN/moving_mean/Read/ReadVariableOp6block_2_project_BN/moving_variance/Read/ReadVariableOp)block_3_expand/kernel/Read/ReadVariableOp*block_3_expand_BN/beta/Read/ReadVariableOp+block_3_expand_BN/gamma/Read/ReadVariableOp1block_3_expand_BN/moving_mean/Read/ReadVariableOp5block_3_expand_BN/moving_variance/Read/ReadVariableOp<expanded_conv_depthwise/depthwise_kernel/Read/ReadVariableOp6block_3_depthwise/depthwise_kernel/Read/ReadVariableOp-block_3_depthwise_BN/beta/Read/ReadVariableOp.block_3_depthwise_BN/gamma/Read/ReadVariableOp4block_3_depthwise_BN/moving_mean/Read/ReadVariableOp8block_3_depthwise_BN/moving_variance/Read/ReadVariableOp*block_3_project/kernel/Read/ReadVariableOp+block_3_project_BN/beta/Read/ReadVariableOp,block_3_project_BN/gamma/Read/ReadVariableOp2block_3_project_BN/moving_mean/Read/ReadVariableOp6block_3_project_BN/moving_variance/Read/ReadVariableOp)block_4_expand/kernel/Read/ReadVariableOp*block_4_expand_BN/beta/Read/ReadVariableOp+block_4_expand_BN/gamma/Read/ReadVariableOp1block_4_expand_BN/moving_mean/Read/ReadVariableOp5block_4_expand_BN/moving_variance/Read/ReadVariableOp6block_4_depthwise/depthwise_kernel/Read/ReadVariableOp-block_4_depthwise_BN/beta/Read/ReadVariableOp.block_4_depthwise_BN/gamma/Read/ReadVariableOp4block_4_depthwise_BN/moving_mean/Read/ReadVariableOp8block_4_depthwise_BN/moving_variance/Read/ReadVariableOp*block_4_project/kernel/Read/ReadVariableOp+block_4_project_BN/beta/Read/ReadVariableOp,block_4_project_BN/gamma/Read/ReadVariableOp2block_4_project_BN/moving_mean/Read/ReadVariableOp6block_4_project_BN/moving_variance/Read/ReadVariableOp3expanded_conv_depthwise_BN/beta/Read/ReadVariableOp4expanded_conv_depthwise_BN/gamma/Read/ReadVariableOp:expanded_conv_depthwise_BN/moving_mean/Read/ReadVariableOp>expanded_conv_depthwise_BN/moving_variance/Read/ReadVariableOp)block_5_expand/kernel/Read/ReadVariableOp*block_5_expand_BN/beta/Read/ReadVariableOp+block_5_expand_BN/gamma/Read/ReadVariableOp1block_5_expand_BN/moving_mean/Read/ReadVariableOp5block_5_expand_BN/moving_variance/Read/ReadVariableOp6block_5_depthwise/depthwise_kernel/Read/ReadVariableOp-block_5_depthwise_BN/beta/Read/ReadVariableOp.block_5_depthwise_BN/gamma/Read/ReadVariableOp4block_5_depthwise_BN/moving_mean/Read/ReadVariableOp8block_5_depthwise_BN/moving_variance/Read/ReadVariableOp*block_5_project/kernel/Read/ReadVariableOp+block_5_project_BN/beta/Read/ReadVariableOp,block_5_project_BN/gamma/Read/ReadVariableOp2block_5_project_BN/moving_mean/Read/ReadVariableOp6block_5_project_BN/moving_variance/Read/ReadVariableOp)block_6_expand/kernel/Read/ReadVariableOp*block_6_expand_BN/beta/Read/ReadVariableOp+block_6_expand_BN/gamma/Read/ReadVariableOp1block_6_expand_BN/moving_mean/Read/ReadVariableOp5block_6_expand_BN/moving_variance/Read/ReadVariableOp6block_6_depthwise/depthwise_kernel/Read/ReadVariableOp-block_6_depthwise_BN/beta/Read/ReadVariableOp.block_6_depthwise_BN/gamma/Read/ReadVariableOp4block_6_depthwise_BN/moving_mean/Read/ReadVariableOp8block_6_depthwise_BN/moving_variance/Read/ReadVariableOp0expanded_conv_project/kernel/Read/ReadVariableOp*block_6_project/kernel/Read/ReadVariableOp+block_6_project_BN/beta/Read/ReadVariableOp,block_6_project_BN/gamma/Read/ReadVariableOp2block_6_project_BN/moving_mean/Read/ReadVariableOp6block_6_project_BN/moving_variance/Read/ReadVariableOp)block_7_expand/kernel/Read/ReadVariableOp*block_7_expand_BN/beta/Read/ReadVariableOp+block_7_expand_BN/gamma/Read/ReadVariableOp1block_7_expand_BN/moving_mean/Read/ReadVariableOp5block_7_expand_BN/moving_variance/Read/ReadVariableOp6block_7_depthwise/depthwise_kernel/Read/ReadVariableOp-block_7_depthwise_BN/beta/Read/ReadVariableOp.block_7_depthwise_BN/gamma/Read/ReadVariableOp4block_7_depthwise_BN/moving_mean/Read/ReadVariableOp8block_7_depthwise_BN/moving_variance/Read/ReadVariableOp*block_7_project/kernel/Read/ReadVariableOp+block_7_project_BN/beta/Read/ReadVariableOp,block_7_project_BN/gamma/Read/ReadVariableOp2block_7_project_BN/moving_mean/Read/ReadVariableOp6block_7_project_BN/moving_variance/Read/ReadVariableOp)block_8_expand/kernel/Read/ReadVariableOp*block_8_expand_BN/beta/Read/ReadVariableOp+block_8_expand_BN/gamma/Read/ReadVariableOp1block_8_expand_BN/moving_mean/Read/ReadVariableOp5block_8_expand_BN/moving_variance/Read/ReadVariableOp1expanded_conv_project_BN/beta/Read/ReadVariableOp2expanded_conv_project_BN/gamma/Read/ReadVariableOp8expanded_conv_project_BN/moving_mean/Read/ReadVariableOp<expanded_conv_project_BN/moving_variance/Read/ReadVariableOp6block_8_depthwise/depthwise_kernel/Read/ReadVariableOp-block_8_depthwise_BN/beta/Read/ReadVariableOp.block_8_depthwise_BN/gamma/Read/ReadVariableOp4block_8_depthwise_BN/moving_mean/Read/ReadVariableOp8block_8_depthwise_BN/moving_variance/Read/ReadVariableOp*block_8_project/kernel/Read/ReadVariableOp+block_8_project_BN/beta/Read/ReadVariableOp,block_8_project_BN/gamma/Read/ReadVariableOp2block_8_project_BN/moving_mean/Read/ReadVariableOp6block_8_project_BN/moving_variance/Read/ReadVariableOp)block_9_expand/kernel/Read/ReadVariableOp*block_9_expand_BN/beta/Read/ReadVariableOp+block_9_expand_BN/gamma/Read/ReadVariableOp1block_9_expand_BN/moving_mean/Read/ReadVariableOp5block_9_expand_BN/moving_variance/Read/ReadVariableOp6block_9_depthwise/depthwise_kernel/Read/ReadVariableOp-block_9_depthwise_BN/beta/Read/ReadVariableOp.block_9_depthwise_BN/gamma/Read/ReadVariableOp4block_9_depthwise_BN/moving_mean/Read/ReadVariableOp8block_9_depthwise_BN/moving_variance/Read/ReadVariableOp*block_9_project/kernel/Read/ReadVariableOp+block_9_project_BN/beta/Read/ReadVariableOp,block_9_project_BN/gamma/Read/ReadVariableOp2block_9_project_BN/moving_mean/Read/ReadVariableOp6block_9_project_BN/moving_variance/Read/ReadVariableOp)block_1_expand/kernel/Read/ReadVariableOp*block_10_expand/kernel/Read/ReadVariableOp+block_10_expand_BN/beta/Read/ReadVariableOp,block_10_expand_BN/gamma/Read/ReadVariableOp2block_10_expand_BN/moving_mean/Read/ReadVariableOp6block_10_expand_BN/moving_variance/Read/ReadVariableOp7block_10_depthwise/depthwise_kernel/Read/ReadVariableOp.block_10_depthwise_BN/beta/Read/ReadVariableOp/block_10_depthwise_BN/gamma/Read/ReadVariableOp5block_10_depthwise_BN/moving_mean/Read/ReadVariableOp9block_10_depthwise_BN/moving_variance/Read/ReadVariableOp+block_10_project/kernel/Read/ReadVariableOp,block_10_project_BN/beta/Read/ReadVariableOp-block_10_project_BN/gamma/Read/ReadVariableOp3block_10_project_BN/moving_mean/Read/ReadVariableOp7block_10_project_BN/moving_variance/Read/ReadVariableOp*block_11_expand/kernel/Read/ReadVariableOp+block_11_expand_BN/beta/Read/ReadVariableOp,block_11_expand_BN/gamma/Read/ReadVariableOp2block_11_expand_BN/moving_mean/Read/ReadVariableOp6block_11_expand_BN/moving_variance/Read/ReadVariableOp7block_11_depthwise/depthwise_kernel/Read/ReadVariableOp.block_11_depthwise_BN/beta/Read/ReadVariableOp/block_11_depthwise_BN/gamma/Read/ReadVariableOp5block_11_depthwise_BN/moving_mean/Read/ReadVariableOp9block_11_depthwise_BN/moving_variance/Read/ReadVariableOp*block_1_expand_BN/beta/Read/ReadVariableOp+block_1_expand_BN/gamma/Read/ReadVariableOp1block_1_expand_BN/moving_mean/Read/ReadVariableOp5block_1_expand_BN/moving_variance/Read/ReadVariableOp+block_11_project/kernel/Read/ReadVariableOp,block_11_project_BN/beta/Read/ReadVariableOp-block_11_project_BN/gamma/Read/ReadVariableOp3block_11_project_BN/moving_mean/Read/ReadVariableOp7block_11_project_BN/moving_variance/Read/ReadVariableOp*block_12_expand/kernel/Read/ReadVariableOp+block_12_expand_BN/beta/Read/ReadVariableOp,block_12_expand_BN/gamma/Read/ReadVariableOp2block_12_expand_BN/moving_mean/Read/ReadVariableOp6block_12_expand_BN/moving_variance/Read/ReadVariableOp7block_12_depthwise/depthwise_kernel/Read/ReadVariableOp.block_12_depthwise_BN/beta/Read/ReadVariableOp/block_12_depthwise_BN/gamma/Read/ReadVariableOp5block_12_depthwise_BN/moving_mean/Read/ReadVariableOp9block_12_depthwise_BN/moving_variance/Read/ReadVariableOp+block_12_project/kernel/Read/ReadVariableOp,block_12_project_BN/beta/Read/ReadVariableOp-block_12_project_BN/gamma/Read/ReadVariableOp3block_12_project_BN/moving_mean/Read/ReadVariableOp7block_12_project_BN/moving_variance/Read/ReadVariableOp*block_13_expand/kernel/Read/ReadVariableOp+block_13_expand_BN/beta/Read/ReadVariableOp,block_13_expand_BN/gamma/Read/ReadVariableOp2block_13_expand_BN/moving_mean/Read/ReadVariableOp6block_13_expand_BN/moving_variance/Read/ReadVariableOp6block_1_depthwise/depthwise_kernel/Read/ReadVariableOp7block_13_depthwise/depthwise_kernel/Read/ReadVariableOp.block_13_depthwise_BN/beta/Read/ReadVariableOp/block_13_depthwise_BN/gamma/Read/ReadVariableOp5block_13_depthwise_BN/moving_mean/Read/ReadVariableOp9block_13_depthwise_BN/moving_variance/Read/ReadVariableOp+block_13_project/kernel/Read/ReadVariableOp,block_13_project_BN/beta/Read/ReadVariableOp-block_13_project_BN/gamma/Read/ReadVariableOp3block_13_project_BN/moving_mean/Read/ReadVariableOp7block_13_project_BN/moving_variance/Read/ReadVariableOp*block_14_expand/kernel/Read/ReadVariableOp+block_14_expand_BN/beta/Read/ReadVariableOp,block_14_expand_BN/gamma/Read/ReadVariableOp2block_14_expand_BN/moving_mean/Read/ReadVariableOp6block_14_expand_BN/moving_variance/Read/ReadVariableOp7block_14_depthwise/depthwise_kernel/Read/ReadVariableOp.block_14_depthwise_BN/beta/Read/ReadVariableOp/block_14_depthwise_BN/gamma/Read/ReadVariableOp5block_14_depthwise_BN/moving_mean/Read/ReadVariableOp9block_14_depthwise_BN/moving_variance/Read/ReadVariableOp+block_14_project/kernel/Read/ReadVariableOp,block_14_project_BN/beta/Read/ReadVariableOp-block_14_project_BN/gamma/Read/ReadVariableOp3block_14_project_BN/moving_mean/Read/ReadVariableOp7block_14_project_BN/moving_variance/Read/ReadVariableOp-block_1_depthwise_BN/beta/Read/ReadVariableOp.block_1_depthwise_BN/gamma/Read/ReadVariableOp4block_1_depthwise_BN/moving_mean/Read/ReadVariableOp8block_1_depthwise_BN/moving_variance/Read/ReadVariableOp*block_15_expand/kernel/Read/ReadVariableOp+block_15_expand_BN/beta/Read/ReadVariableOp,block_15_expand_BN/gamma/Read/ReadVariableOp2block_15_expand_BN/moving_mean/Read/ReadVariableOp6block_15_expand_BN/moving_variance/Read/ReadVariableOp7block_15_depthwise/depthwise_kernel/Read/ReadVariableOp.block_15_depthwise_BN/beta/Read/ReadVariableOp/block_15_depthwise_BN/gamma/Read/ReadVariableOp5block_15_depthwise_BN/moving_mean/Read/ReadVariableOp9block_15_depthwise_BN/moving_variance/Read/ReadVariableOp+block_15_project/kernel/Read/ReadVariableOp,block_15_project_BN/beta/Read/ReadVariableOp-block_15_project_BN/gamma/Read/ReadVariableOp3block_15_project_BN/moving_mean/Read/ReadVariableOp7block_15_project_BN/moving_variance/Read/ReadVariableOp*block_16_expand/kernel/Read/ReadVariableOp+block_16_expand_BN/beta/Read/ReadVariableOp,block_16_expand_BN/gamma/Read/ReadVariableOp2block_16_expand_BN/moving_mean/Read/ReadVariableOp6block_16_expand_BN/moving_variance/Read/ReadVariableOp7block_16_depthwise/depthwise_kernel/Read/ReadVariableOp.block_16_depthwise_BN/beta/Read/ReadVariableOp/block_16_depthwise_BN/gamma/Read/ReadVariableOp5block_16_depthwise_BN/moving_mean/Read/ReadVariableOp9block_16_depthwise_BN/moving_variance/Read/ReadVariableOp*�
dtypes�
�2�
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�z
save/RestoreV2/tensor_namesConst"/device:CPU:0*�y
value�yB�y�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-100/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-101/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-101/gamma/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-101/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-101/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-102/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-103/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-103/gamma/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-103/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-103/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-20/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-23/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-23/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-23/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-25/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-25/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-25/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-26/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-27/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-27/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-27/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-29/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-29/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-29/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-31/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-31/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-31/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-31/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-32/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-33/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-33/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-33/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-33/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-35/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-35/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-35/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-35/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-37/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-37/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-37/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-37/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-38/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-39/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-39/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-39/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-39/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-41/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-41/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-41/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-41/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-42/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-43/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-43/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-43/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-43/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-44/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-45/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-45/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-45/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-45/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-46/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-47/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-47/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-47/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-47/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-48/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-49/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-49/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-49/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-49/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-50/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-51/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-51/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-51/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-51/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-52/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-53/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-53/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-53/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-53/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-54/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-55/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-55/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-55/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-55/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-56/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-57/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-57/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-57/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-57/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-58/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-59/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-59/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-59/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-59/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-60/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-61/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-61/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-61/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-61/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-62/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-63/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-63/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-63/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-63/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-64/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-65/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-65/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-65/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-65/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-66/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-67/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-67/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-67/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-67/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-68/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-69/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-69/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-69/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-69/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-70/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-71/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-71/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-71/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-71/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-72/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-73/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-73/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-73/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-73/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-74/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-75/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-75/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-75/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-75/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-76/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-77/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-77/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-77/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-77/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-78/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-79/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-79/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-79/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-79/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-80/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-81/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-81/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-81/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-81/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-82/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-83/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-83/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-83/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-83/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-84/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-85/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-85/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-85/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-85/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-86/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-87/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-87/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-87/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-87/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-88/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-89/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-89/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-89/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-89/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-90/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-91/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-91/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-91/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-91/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-92/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-93/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-93/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-93/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-93/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-94/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-95/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-95/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-95/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-95/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-96/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-97/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-97/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-97/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-97/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-98/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-99/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-99/gamma/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-99/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-99/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes	
:�
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:�
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*�
dtypes�
�2�*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
L
save/IdentityIdentitysave/RestoreV2*
T0*
_output_shapes
:
S
save/AssignVariableOpAssignVariableOpConv1/kernelsave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:1*
T0*
_output_shapes
:
X
save/AssignVariableOp_1AssignVariableOpbn_Conv1/betasave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:2*
T0*
_output_shapes
:
Y
save/AssignVariableOp_2AssignVariableOpbn_Conv1/gammasave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:3*
T0*
_output_shapes
:
_
save/AssignVariableOp_3AssignVariableOpbn_Conv1/moving_meansave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:4*
T0*
_output_shapes
:
c
save/AssignVariableOp_4AssignVariableOpbn_Conv1/moving_variancesave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:5*
T0*
_output_shapes
:
a
save/AssignVariableOp_5AssignVariableOpblock_1_project/kernelsave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:6*
T0*
_output_shapes
:
b
save/AssignVariableOp_6AssignVariableOpblock_16_project/kernelsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:7*
T0*
_output_shapes
:
c
save/AssignVariableOp_7AssignVariableOpblock_16_project_BN/betasave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:8*
T0*
_output_shapes
:
d
save/AssignVariableOp_8AssignVariableOpblock_16_project_BN/gammasave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:9*
T0*
_output_shapes
:
j
save/AssignVariableOp_9AssignVariableOpblock_16_project_BN/moving_meansave/Identity_9*
dtype0
R
save/Identity_10Identitysave/RestoreV2:10*
T0*
_output_shapes
:
p
save/AssignVariableOp_10AssignVariableOp#block_16_project_BN/moving_variancesave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:11*
T0*
_output_shapes
:
Z
save/AssignVariableOp_11AssignVariableOpConv_1/kernelsave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:12*
T0*
_output_shapes
:
[
save/AssignVariableOp_12AssignVariableOpConv_1_bn/betasave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:13*
T0*
_output_shapes
:
\
save/AssignVariableOp_13AssignVariableOpConv_1_bn/gammasave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:14*
T0*
_output_shapes
:
b
save/AssignVariableOp_14AssignVariableOpConv_1_bn/moving_meansave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:15*
T0*
_output_shapes
:
f
save/AssignVariableOp_15AssignVariableOpConv_1_bn/moving_variancesave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:16*
T0*
_output_shapes
:
d
save/AssignVariableOp_16AssignVariableOpblock_1_project_BN/betasave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:17*
T0*
_output_shapes
:
e
save/AssignVariableOp_17AssignVariableOpblock_1_project_BN/gammasave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:18*
T0*
_output_shapes
:
k
save/AssignVariableOp_18AssignVariableOpblock_1_project_BN/moving_meansave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:19*
T0*
_output_shapes
:
o
save/AssignVariableOp_19AssignVariableOp"block_1_project_BN/moving_variancesave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:20*
T0*
_output_shapes
:
b
save/AssignVariableOp_20AssignVariableOpblock_2_expand/kernelsave/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:21*
T0*
_output_shapes
:
c
save/AssignVariableOp_21AssignVariableOpblock_2_expand_BN/betasave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:22*
T0*
_output_shapes
:
d
save/AssignVariableOp_22AssignVariableOpblock_2_expand_BN/gammasave/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:23*
T0*
_output_shapes
:
j
save/AssignVariableOp_23AssignVariableOpblock_2_expand_BN/moving_meansave/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:24*
T0*
_output_shapes
:
n
save/AssignVariableOp_24AssignVariableOp!block_2_expand_BN/moving_variancesave/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:25*
T0*
_output_shapes
:
o
save/AssignVariableOp_25AssignVariableOp"block_2_depthwise/depthwise_kernelsave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:26*
T0*
_output_shapes
:
f
save/AssignVariableOp_26AssignVariableOpblock_2_depthwise_BN/betasave/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:27*
T0*
_output_shapes
:
g
save/AssignVariableOp_27AssignVariableOpblock_2_depthwise_BN/gammasave/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:28*
T0*
_output_shapes
:
m
save/AssignVariableOp_28AssignVariableOp block_2_depthwise_BN/moving_meansave/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:29*
T0*
_output_shapes
:
q
save/AssignVariableOp_29AssignVariableOp$block_2_depthwise_BN/moving_variancesave/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:30*
T0*
_output_shapes
:
c
save/AssignVariableOp_30AssignVariableOpblock_2_project/kernelsave/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:31*
T0*
_output_shapes
:
d
save/AssignVariableOp_31AssignVariableOpblock_2_project_BN/betasave/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:32*
T0*
_output_shapes
:
e
save/AssignVariableOp_32AssignVariableOpblock_2_project_BN/gammasave/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:33*
T0*
_output_shapes
:
k
save/AssignVariableOp_33AssignVariableOpblock_2_project_BN/moving_meansave/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:34*
T0*
_output_shapes
:
o
save/AssignVariableOp_34AssignVariableOp"block_2_project_BN/moving_variancesave/Identity_34*
dtype0
R
save/Identity_35Identitysave/RestoreV2:35*
T0*
_output_shapes
:
b
save/AssignVariableOp_35AssignVariableOpblock_3_expand/kernelsave/Identity_35*
dtype0
R
save/Identity_36Identitysave/RestoreV2:36*
T0*
_output_shapes
:
c
save/AssignVariableOp_36AssignVariableOpblock_3_expand_BN/betasave/Identity_36*
dtype0
R
save/Identity_37Identitysave/RestoreV2:37*
T0*
_output_shapes
:
d
save/AssignVariableOp_37AssignVariableOpblock_3_expand_BN/gammasave/Identity_37*
dtype0
R
save/Identity_38Identitysave/RestoreV2:38*
T0*
_output_shapes
:
j
save/AssignVariableOp_38AssignVariableOpblock_3_expand_BN/moving_meansave/Identity_38*
dtype0
R
save/Identity_39Identitysave/RestoreV2:39*
T0*
_output_shapes
:
n
save/AssignVariableOp_39AssignVariableOp!block_3_expand_BN/moving_variancesave/Identity_39*
dtype0
R
save/Identity_40Identitysave/RestoreV2:40*
T0*
_output_shapes
:
u
save/AssignVariableOp_40AssignVariableOp(expanded_conv_depthwise/depthwise_kernelsave/Identity_40*
dtype0
R
save/Identity_41Identitysave/RestoreV2:41*
T0*
_output_shapes
:
o
save/AssignVariableOp_41AssignVariableOp"block_3_depthwise/depthwise_kernelsave/Identity_41*
dtype0
R
save/Identity_42Identitysave/RestoreV2:42*
T0*
_output_shapes
:
f
save/AssignVariableOp_42AssignVariableOpblock_3_depthwise_BN/betasave/Identity_42*
dtype0
R
save/Identity_43Identitysave/RestoreV2:43*
T0*
_output_shapes
:
g
save/AssignVariableOp_43AssignVariableOpblock_3_depthwise_BN/gammasave/Identity_43*
dtype0
R
save/Identity_44Identitysave/RestoreV2:44*
T0*
_output_shapes
:
m
save/AssignVariableOp_44AssignVariableOp block_3_depthwise_BN/moving_meansave/Identity_44*
dtype0
R
save/Identity_45Identitysave/RestoreV2:45*
T0*
_output_shapes
:
q
save/AssignVariableOp_45AssignVariableOp$block_3_depthwise_BN/moving_variancesave/Identity_45*
dtype0
R
save/Identity_46Identitysave/RestoreV2:46*
T0*
_output_shapes
:
c
save/AssignVariableOp_46AssignVariableOpblock_3_project/kernelsave/Identity_46*
dtype0
R
save/Identity_47Identitysave/RestoreV2:47*
T0*
_output_shapes
:
d
save/AssignVariableOp_47AssignVariableOpblock_3_project_BN/betasave/Identity_47*
dtype0
R
save/Identity_48Identitysave/RestoreV2:48*
T0*
_output_shapes
:
e
save/AssignVariableOp_48AssignVariableOpblock_3_project_BN/gammasave/Identity_48*
dtype0
R
save/Identity_49Identitysave/RestoreV2:49*
T0*
_output_shapes
:
k
save/AssignVariableOp_49AssignVariableOpblock_3_project_BN/moving_meansave/Identity_49*
dtype0
R
save/Identity_50Identitysave/RestoreV2:50*
T0*
_output_shapes
:
o
save/AssignVariableOp_50AssignVariableOp"block_3_project_BN/moving_variancesave/Identity_50*
dtype0
R
save/Identity_51Identitysave/RestoreV2:51*
T0*
_output_shapes
:
b
save/AssignVariableOp_51AssignVariableOpblock_4_expand/kernelsave/Identity_51*
dtype0
R
save/Identity_52Identitysave/RestoreV2:52*
T0*
_output_shapes
:
c
save/AssignVariableOp_52AssignVariableOpblock_4_expand_BN/betasave/Identity_52*
dtype0
R
save/Identity_53Identitysave/RestoreV2:53*
T0*
_output_shapes
:
d
save/AssignVariableOp_53AssignVariableOpblock_4_expand_BN/gammasave/Identity_53*
dtype0
R
save/Identity_54Identitysave/RestoreV2:54*
T0*
_output_shapes
:
j
save/AssignVariableOp_54AssignVariableOpblock_4_expand_BN/moving_meansave/Identity_54*
dtype0
R
save/Identity_55Identitysave/RestoreV2:55*
T0*
_output_shapes
:
n
save/AssignVariableOp_55AssignVariableOp!block_4_expand_BN/moving_variancesave/Identity_55*
dtype0
R
save/Identity_56Identitysave/RestoreV2:56*
T0*
_output_shapes
:
o
save/AssignVariableOp_56AssignVariableOp"block_4_depthwise/depthwise_kernelsave/Identity_56*
dtype0
R
save/Identity_57Identitysave/RestoreV2:57*
T0*
_output_shapes
:
f
save/AssignVariableOp_57AssignVariableOpblock_4_depthwise_BN/betasave/Identity_57*
dtype0
R
save/Identity_58Identitysave/RestoreV2:58*
T0*
_output_shapes
:
g
save/AssignVariableOp_58AssignVariableOpblock_4_depthwise_BN/gammasave/Identity_58*
dtype0
R
save/Identity_59Identitysave/RestoreV2:59*
T0*
_output_shapes
:
m
save/AssignVariableOp_59AssignVariableOp block_4_depthwise_BN/moving_meansave/Identity_59*
dtype0
R
save/Identity_60Identitysave/RestoreV2:60*
T0*
_output_shapes
:
q
save/AssignVariableOp_60AssignVariableOp$block_4_depthwise_BN/moving_variancesave/Identity_60*
dtype0
R
save/Identity_61Identitysave/RestoreV2:61*
T0*
_output_shapes
:
c
save/AssignVariableOp_61AssignVariableOpblock_4_project/kernelsave/Identity_61*
dtype0
R
save/Identity_62Identitysave/RestoreV2:62*
T0*
_output_shapes
:
d
save/AssignVariableOp_62AssignVariableOpblock_4_project_BN/betasave/Identity_62*
dtype0
R
save/Identity_63Identitysave/RestoreV2:63*
T0*
_output_shapes
:
e
save/AssignVariableOp_63AssignVariableOpblock_4_project_BN/gammasave/Identity_63*
dtype0
R
save/Identity_64Identitysave/RestoreV2:64*
T0*
_output_shapes
:
k
save/AssignVariableOp_64AssignVariableOpblock_4_project_BN/moving_meansave/Identity_64*
dtype0
R
save/Identity_65Identitysave/RestoreV2:65*
T0*
_output_shapes
:
o
save/AssignVariableOp_65AssignVariableOp"block_4_project_BN/moving_variancesave/Identity_65*
dtype0
R
save/Identity_66Identitysave/RestoreV2:66*
T0*
_output_shapes
:
l
save/AssignVariableOp_66AssignVariableOpexpanded_conv_depthwise_BN/betasave/Identity_66*
dtype0
R
save/Identity_67Identitysave/RestoreV2:67*
T0*
_output_shapes
:
m
save/AssignVariableOp_67AssignVariableOp expanded_conv_depthwise_BN/gammasave/Identity_67*
dtype0
R
save/Identity_68Identitysave/RestoreV2:68*
T0*
_output_shapes
:
s
save/AssignVariableOp_68AssignVariableOp&expanded_conv_depthwise_BN/moving_meansave/Identity_68*
dtype0
R
save/Identity_69Identitysave/RestoreV2:69*
T0*
_output_shapes
:
w
save/AssignVariableOp_69AssignVariableOp*expanded_conv_depthwise_BN/moving_variancesave/Identity_69*
dtype0
R
save/Identity_70Identitysave/RestoreV2:70*
T0*
_output_shapes
:
b
save/AssignVariableOp_70AssignVariableOpblock_5_expand/kernelsave/Identity_70*
dtype0
R
save/Identity_71Identitysave/RestoreV2:71*
T0*
_output_shapes
:
c
save/AssignVariableOp_71AssignVariableOpblock_5_expand_BN/betasave/Identity_71*
dtype0
R
save/Identity_72Identitysave/RestoreV2:72*
T0*
_output_shapes
:
d
save/AssignVariableOp_72AssignVariableOpblock_5_expand_BN/gammasave/Identity_72*
dtype0
R
save/Identity_73Identitysave/RestoreV2:73*
T0*
_output_shapes
:
j
save/AssignVariableOp_73AssignVariableOpblock_5_expand_BN/moving_meansave/Identity_73*
dtype0
R
save/Identity_74Identitysave/RestoreV2:74*
T0*
_output_shapes
:
n
save/AssignVariableOp_74AssignVariableOp!block_5_expand_BN/moving_variancesave/Identity_74*
dtype0
R
save/Identity_75Identitysave/RestoreV2:75*
T0*
_output_shapes
:
o
save/AssignVariableOp_75AssignVariableOp"block_5_depthwise/depthwise_kernelsave/Identity_75*
dtype0
R
save/Identity_76Identitysave/RestoreV2:76*
T0*
_output_shapes
:
f
save/AssignVariableOp_76AssignVariableOpblock_5_depthwise_BN/betasave/Identity_76*
dtype0
R
save/Identity_77Identitysave/RestoreV2:77*
T0*
_output_shapes
:
g
save/AssignVariableOp_77AssignVariableOpblock_5_depthwise_BN/gammasave/Identity_77*
dtype0
R
save/Identity_78Identitysave/RestoreV2:78*
T0*
_output_shapes
:
m
save/AssignVariableOp_78AssignVariableOp block_5_depthwise_BN/moving_meansave/Identity_78*
dtype0
R
save/Identity_79Identitysave/RestoreV2:79*
T0*
_output_shapes
:
q
save/AssignVariableOp_79AssignVariableOp$block_5_depthwise_BN/moving_variancesave/Identity_79*
dtype0
R
save/Identity_80Identitysave/RestoreV2:80*
T0*
_output_shapes
:
c
save/AssignVariableOp_80AssignVariableOpblock_5_project/kernelsave/Identity_80*
dtype0
R
save/Identity_81Identitysave/RestoreV2:81*
T0*
_output_shapes
:
d
save/AssignVariableOp_81AssignVariableOpblock_5_project_BN/betasave/Identity_81*
dtype0
R
save/Identity_82Identitysave/RestoreV2:82*
T0*
_output_shapes
:
e
save/AssignVariableOp_82AssignVariableOpblock_5_project_BN/gammasave/Identity_82*
dtype0
R
save/Identity_83Identitysave/RestoreV2:83*
T0*
_output_shapes
:
k
save/AssignVariableOp_83AssignVariableOpblock_5_project_BN/moving_meansave/Identity_83*
dtype0
R
save/Identity_84Identitysave/RestoreV2:84*
T0*
_output_shapes
:
o
save/AssignVariableOp_84AssignVariableOp"block_5_project_BN/moving_variancesave/Identity_84*
dtype0
R
save/Identity_85Identitysave/RestoreV2:85*
T0*
_output_shapes
:
b
save/AssignVariableOp_85AssignVariableOpblock_6_expand/kernelsave/Identity_85*
dtype0
R
save/Identity_86Identitysave/RestoreV2:86*
T0*
_output_shapes
:
c
save/AssignVariableOp_86AssignVariableOpblock_6_expand_BN/betasave/Identity_86*
dtype0
R
save/Identity_87Identitysave/RestoreV2:87*
T0*
_output_shapes
:
d
save/AssignVariableOp_87AssignVariableOpblock_6_expand_BN/gammasave/Identity_87*
dtype0
R
save/Identity_88Identitysave/RestoreV2:88*
T0*
_output_shapes
:
j
save/AssignVariableOp_88AssignVariableOpblock_6_expand_BN/moving_meansave/Identity_88*
dtype0
R
save/Identity_89Identitysave/RestoreV2:89*
T0*
_output_shapes
:
n
save/AssignVariableOp_89AssignVariableOp!block_6_expand_BN/moving_variancesave/Identity_89*
dtype0
R
save/Identity_90Identitysave/RestoreV2:90*
T0*
_output_shapes
:
o
save/AssignVariableOp_90AssignVariableOp"block_6_depthwise/depthwise_kernelsave/Identity_90*
dtype0
R
save/Identity_91Identitysave/RestoreV2:91*
T0*
_output_shapes
:
f
save/AssignVariableOp_91AssignVariableOpblock_6_depthwise_BN/betasave/Identity_91*
dtype0
R
save/Identity_92Identitysave/RestoreV2:92*
T0*
_output_shapes
:
g
save/AssignVariableOp_92AssignVariableOpblock_6_depthwise_BN/gammasave/Identity_92*
dtype0
R
save/Identity_93Identitysave/RestoreV2:93*
T0*
_output_shapes
:
m
save/AssignVariableOp_93AssignVariableOp block_6_depthwise_BN/moving_meansave/Identity_93*
dtype0
R
save/Identity_94Identitysave/RestoreV2:94*
T0*
_output_shapes
:
q
save/AssignVariableOp_94AssignVariableOp$block_6_depthwise_BN/moving_variancesave/Identity_94*
dtype0
R
save/Identity_95Identitysave/RestoreV2:95*
T0*
_output_shapes
:
i
save/AssignVariableOp_95AssignVariableOpexpanded_conv_project/kernelsave/Identity_95*
dtype0
R
save/Identity_96Identitysave/RestoreV2:96*
T0*
_output_shapes
:
c
save/AssignVariableOp_96AssignVariableOpblock_6_project/kernelsave/Identity_96*
dtype0
R
save/Identity_97Identitysave/RestoreV2:97*
T0*
_output_shapes
:
d
save/AssignVariableOp_97AssignVariableOpblock_6_project_BN/betasave/Identity_97*
dtype0
R
save/Identity_98Identitysave/RestoreV2:98*
T0*
_output_shapes
:
e
save/AssignVariableOp_98AssignVariableOpblock_6_project_BN/gammasave/Identity_98*
dtype0
R
save/Identity_99Identitysave/RestoreV2:99*
T0*
_output_shapes
:
k
save/AssignVariableOp_99AssignVariableOpblock_6_project_BN/moving_meansave/Identity_99*
dtype0
T
save/Identity_100Identitysave/RestoreV2:100*
T0*
_output_shapes
:
q
save/AssignVariableOp_100AssignVariableOp"block_6_project_BN/moving_variancesave/Identity_100*
dtype0
T
save/Identity_101Identitysave/RestoreV2:101*
T0*
_output_shapes
:
d
save/AssignVariableOp_101AssignVariableOpblock_7_expand/kernelsave/Identity_101*
dtype0
T
save/Identity_102Identitysave/RestoreV2:102*
T0*
_output_shapes
:
e
save/AssignVariableOp_102AssignVariableOpblock_7_expand_BN/betasave/Identity_102*
dtype0
T
save/Identity_103Identitysave/RestoreV2:103*
T0*
_output_shapes
:
f
save/AssignVariableOp_103AssignVariableOpblock_7_expand_BN/gammasave/Identity_103*
dtype0
T
save/Identity_104Identitysave/RestoreV2:104*
T0*
_output_shapes
:
l
save/AssignVariableOp_104AssignVariableOpblock_7_expand_BN/moving_meansave/Identity_104*
dtype0
T
save/Identity_105Identitysave/RestoreV2:105*
T0*
_output_shapes
:
p
save/AssignVariableOp_105AssignVariableOp!block_7_expand_BN/moving_variancesave/Identity_105*
dtype0
T
save/Identity_106Identitysave/RestoreV2:106*
T0*
_output_shapes
:
q
save/AssignVariableOp_106AssignVariableOp"block_7_depthwise/depthwise_kernelsave/Identity_106*
dtype0
T
save/Identity_107Identitysave/RestoreV2:107*
T0*
_output_shapes
:
h
save/AssignVariableOp_107AssignVariableOpblock_7_depthwise_BN/betasave/Identity_107*
dtype0
T
save/Identity_108Identitysave/RestoreV2:108*
T0*
_output_shapes
:
i
save/AssignVariableOp_108AssignVariableOpblock_7_depthwise_BN/gammasave/Identity_108*
dtype0
T
save/Identity_109Identitysave/RestoreV2:109*
T0*
_output_shapes
:
o
save/AssignVariableOp_109AssignVariableOp block_7_depthwise_BN/moving_meansave/Identity_109*
dtype0
T
save/Identity_110Identitysave/RestoreV2:110*
T0*
_output_shapes
:
s
save/AssignVariableOp_110AssignVariableOp$block_7_depthwise_BN/moving_variancesave/Identity_110*
dtype0
T
save/Identity_111Identitysave/RestoreV2:111*
T0*
_output_shapes
:
e
save/AssignVariableOp_111AssignVariableOpblock_7_project/kernelsave/Identity_111*
dtype0
T
save/Identity_112Identitysave/RestoreV2:112*
T0*
_output_shapes
:
f
save/AssignVariableOp_112AssignVariableOpblock_7_project_BN/betasave/Identity_112*
dtype0
T
save/Identity_113Identitysave/RestoreV2:113*
T0*
_output_shapes
:
g
save/AssignVariableOp_113AssignVariableOpblock_7_project_BN/gammasave/Identity_113*
dtype0
T
save/Identity_114Identitysave/RestoreV2:114*
T0*
_output_shapes
:
m
save/AssignVariableOp_114AssignVariableOpblock_7_project_BN/moving_meansave/Identity_114*
dtype0
T
save/Identity_115Identitysave/RestoreV2:115*
T0*
_output_shapes
:
q
save/AssignVariableOp_115AssignVariableOp"block_7_project_BN/moving_variancesave/Identity_115*
dtype0
T
save/Identity_116Identitysave/RestoreV2:116*
T0*
_output_shapes
:
d
save/AssignVariableOp_116AssignVariableOpblock_8_expand/kernelsave/Identity_116*
dtype0
T
save/Identity_117Identitysave/RestoreV2:117*
T0*
_output_shapes
:
e
save/AssignVariableOp_117AssignVariableOpblock_8_expand_BN/betasave/Identity_117*
dtype0
T
save/Identity_118Identitysave/RestoreV2:118*
T0*
_output_shapes
:
f
save/AssignVariableOp_118AssignVariableOpblock_8_expand_BN/gammasave/Identity_118*
dtype0
T
save/Identity_119Identitysave/RestoreV2:119*
T0*
_output_shapes
:
l
save/AssignVariableOp_119AssignVariableOpblock_8_expand_BN/moving_meansave/Identity_119*
dtype0
T
save/Identity_120Identitysave/RestoreV2:120*
T0*
_output_shapes
:
p
save/AssignVariableOp_120AssignVariableOp!block_8_expand_BN/moving_variancesave/Identity_120*
dtype0
T
save/Identity_121Identitysave/RestoreV2:121*
T0*
_output_shapes
:
l
save/AssignVariableOp_121AssignVariableOpexpanded_conv_project_BN/betasave/Identity_121*
dtype0
T
save/Identity_122Identitysave/RestoreV2:122*
T0*
_output_shapes
:
m
save/AssignVariableOp_122AssignVariableOpexpanded_conv_project_BN/gammasave/Identity_122*
dtype0
T
save/Identity_123Identitysave/RestoreV2:123*
T0*
_output_shapes
:
s
save/AssignVariableOp_123AssignVariableOp$expanded_conv_project_BN/moving_meansave/Identity_123*
dtype0
T
save/Identity_124Identitysave/RestoreV2:124*
T0*
_output_shapes
:
w
save/AssignVariableOp_124AssignVariableOp(expanded_conv_project_BN/moving_variancesave/Identity_124*
dtype0
T
save/Identity_125Identitysave/RestoreV2:125*
T0*
_output_shapes
:
q
save/AssignVariableOp_125AssignVariableOp"block_8_depthwise/depthwise_kernelsave/Identity_125*
dtype0
T
save/Identity_126Identitysave/RestoreV2:126*
T0*
_output_shapes
:
h
save/AssignVariableOp_126AssignVariableOpblock_8_depthwise_BN/betasave/Identity_126*
dtype0
T
save/Identity_127Identitysave/RestoreV2:127*
T0*
_output_shapes
:
i
save/AssignVariableOp_127AssignVariableOpblock_8_depthwise_BN/gammasave/Identity_127*
dtype0
T
save/Identity_128Identitysave/RestoreV2:128*
T0*
_output_shapes
:
o
save/AssignVariableOp_128AssignVariableOp block_8_depthwise_BN/moving_meansave/Identity_128*
dtype0
T
save/Identity_129Identitysave/RestoreV2:129*
T0*
_output_shapes
:
s
save/AssignVariableOp_129AssignVariableOp$block_8_depthwise_BN/moving_variancesave/Identity_129*
dtype0
T
save/Identity_130Identitysave/RestoreV2:130*
T0*
_output_shapes
:
e
save/AssignVariableOp_130AssignVariableOpblock_8_project/kernelsave/Identity_130*
dtype0
T
save/Identity_131Identitysave/RestoreV2:131*
T0*
_output_shapes
:
f
save/AssignVariableOp_131AssignVariableOpblock_8_project_BN/betasave/Identity_131*
dtype0
T
save/Identity_132Identitysave/RestoreV2:132*
T0*
_output_shapes
:
g
save/AssignVariableOp_132AssignVariableOpblock_8_project_BN/gammasave/Identity_132*
dtype0
T
save/Identity_133Identitysave/RestoreV2:133*
T0*
_output_shapes
:
m
save/AssignVariableOp_133AssignVariableOpblock_8_project_BN/moving_meansave/Identity_133*
dtype0
T
save/Identity_134Identitysave/RestoreV2:134*
T0*
_output_shapes
:
q
save/AssignVariableOp_134AssignVariableOp"block_8_project_BN/moving_variancesave/Identity_134*
dtype0
T
save/Identity_135Identitysave/RestoreV2:135*
T0*
_output_shapes
:
d
save/AssignVariableOp_135AssignVariableOpblock_9_expand/kernelsave/Identity_135*
dtype0
T
save/Identity_136Identitysave/RestoreV2:136*
T0*
_output_shapes
:
e
save/AssignVariableOp_136AssignVariableOpblock_9_expand_BN/betasave/Identity_136*
dtype0
T
save/Identity_137Identitysave/RestoreV2:137*
T0*
_output_shapes
:
f
save/AssignVariableOp_137AssignVariableOpblock_9_expand_BN/gammasave/Identity_137*
dtype0
T
save/Identity_138Identitysave/RestoreV2:138*
T0*
_output_shapes
:
l
save/AssignVariableOp_138AssignVariableOpblock_9_expand_BN/moving_meansave/Identity_138*
dtype0
T
save/Identity_139Identitysave/RestoreV2:139*
T0*
_output_shapes
:
p
save/AssignVariableOp_139AssignVariableOp!block_9_expand_BN/moving_variancesave/Identity_139*
dtype0
T
save/Identity_140Identitysave/RestoreV2:140*
T0*
_output_shapes
:
q
save/AssignVariableOp_140AssignVariableOp"block_9_depthwise/depthwise_kernelsave/Identity_140*
dtype0
T
save/Identity_141Identitysave/RestoreV2:141*
T0*
_output_shapes
:
h
save/AssignVariableOp_141AssignVariableOpblock_9_depthwise_BN/betasave/Identity_141*
dtype0
T
save/Identity_142Identitysave/RestoreV2:142*
T0*
_output_shapes
:
i
save/AssignVariableOp_142AssignVariableOpblock_9_depthwise_BN/gammasave/Identity_142*
dtype0
T
save/Identity_143Identitysave/RestoreV2:143*
T0*
_output_shapes
:
o
save/AssignVariableOp_143AssignVariableOp block_9_depthwise_BN/moving_meansave/Identity_143*
dtype0
T
save/Identity_144Identitysave/RestoreV2:144*
T0*
_output_shapes
:
s
save/AssignVariableOp_144AssignVariableOp$block_9_depthwise_BN/moving_variancesave/Identity_144*
dtype0
T
save/Identity_145Identitysave/RestoreV2:145*
T0*
_output_shapes
:
e
save/AssignVariableOp_145AssignVariableOpblock_9_project/kernelsave/Identity_145*
dtype0
T
save/Identity_146Identitysave/RestoreV2:146*
T0*
_output_shapes
:
f
save/AssignVariableOp_146AssignVariableOpblock_9_project_BN/betasave/Identity_146*
dtype0
T
save/Identity_147Identitysave/RestoreV2:147*
T0*
_output_shapes
:
g
save/AssignVariableOp_147AssignVariableOpblock_9_project_BN/gammasave/Identity_147*
dtype0
T
save/Identity_148Identitysave/RestoreV2:148*
T0*
_output_shapes
:
m
save/AssignVariableOp_148AssignVariableOpblock_9_project_BN/moving_meansave/Identity_148*
dtype0
T
save/Identity_149Identitysave/RestoreV2:149*
T0*
_output_shapes
:
q
save/AssignVariableOp_149AssignVariableOp"block_9_project_BN/moving_variancesave/Identity_149*
dtype0
T
save/Identity_150Identitysave/RestoreV2:150*
T0*
_output_shapes
:
d
save/AssignVariableOp_150AssignVariableOpblock_1_expand/kernelsave/Identity_150*
dtype0
T
save/Identity_151Identitysave/RestoreV2:151*
T0*
_output_shapes
:
e
save/AssignVariableOp_151AssignVariableOpblock_10_expand/kernelsave/Identity_151*
dtype0
T
save/Identity_152Identitysave/RestoreV2:152*
T0*
_output_shapes
:
f
save/AssignVariableOp_152AssignVariableOpblock_10_expand_BN/betasave/Identity_152*
dtype0
T
save/Identity_153Identitysave/RestoreV2:153*
T0*
_output_shapes
:
g
save/AssignVariableOp_153AssignVariableOpblock_10_expand_BN/gammasave/Identity_153*
dtype0
T
save/Identity_154Identitysave/RestoreV2:154*
T0*
_output_shapes
:
m
save/AssignVariableOp_154AssignVariableOpblock_10_expand_BN/moving_meansave/Identity_154*
dtype0
T
save/Identity_155Identitysave/RestoreV2:155*
T0*
_output_shapes
:
q
save/AssignVariableOp_155AssignVariableOp"block_10_expand_BN/moving_variancesave/Identity_155*
dtype0
T
save/Identity_156Identitysave/RestoreV2:156*
T0*
_output_shapes
:
r
save/AssignVariableOp_156AssignVariableOp#block_10_depthwise/depthwise_kernelsave/Identity_156*
dtype0
T
save/Identity_157Identitysave/RestoreV2:157*
T0*
_output_shapes
:
i
save/AssignVariableOp_157AssignVariableOpblock_10_depthwise_BN/betasave/Identity_157*
dtype0
T
save/Identity_158Identitysave/RestoreV2:158*
T0*
_output_shapes
:
j
save/AssignVariableOp_158AssignVariableOpblock_10_depthwise_BN/gammasave/Identity_158*
dtype0
T
save/Identity_159Identitysave/RestoreV2:159*
T0*
_output_shapes
:
p
save/AssignVariableOp_159AssignVariableOp!block_10_depthwise_BN/moving_meansave/Identity_159*
dtype0
T
save/Identity_160Identitysave/RestoreV2:160*
T0*
_output_shapes
:
t
save/AssignVariableOp_160AssignVariableOp%block_10_depthwise_BN/moving_variancesave/Identity_160*
dtype0
T
save/Identity_161Identitysave/RestoreV2:161*
T0*
_output_shapes
:
f
save/AssignVariableOp_161AssignVariableOpblock_10_project/kernelsave/Identity_161*
dtype0
T
save/Identity_162Identitysave/RestoreV2:162*
T0*
_output_shapes
:
g
save/AssignVariableOp_162AssignVariableOpblock_10_project_BN/betasave/Identity_162*
dtype0
T
save/Identity_163Identitysave/RestoreV2:163*
T0*
_output_shapes
:
h
save/AssignVariableOp_163AssignVariableOpblock_10_project_BN/gammasave/Identity_163*
dtype0
T
save/Identity_164Identitysave/RestoreV2:164*
T0*
_output_shapes
:
n
save/AssignVariableOp_164AssignVariableOpblock_10_project_BN/moving_meansave/Identity_164*
dtype0
T
save/Identity_165Identitysave/RestoreV2:165*
T0*
_output_shapes
:
r
save/AssignVariableOp_165AssignVariableOp#block_10_project_BN/moving_variancesave/Identity_165*
dtype0
T
save/Identity_166Identitysave/RestoreV2:166*
T0*
_output_shapes
:
e
save/AssignVariableOp_166AssignVariableOpblock_11_expand/kernelsave/Identity_166*
dtype0
T
save/Identity_167Identitysave/RestoreV2:167*
T0*
_output_shapes
:
f
save/AssignVariableOp_167AssignVariableOpblock_11_expand_BN/betasave/Identity_167*
dtype0
T
save/Identity_168Identitysave/RestoreV2:168*
T0*
_output_shapes
:
g
save/AssignVariableOp_168AssignVariableOpblock_11_expand_BN/gammasave/Identity_168*
dtype0
T
save/Identity_169Identitysave/RestoreV2:169*
T0*
_output_shapes
:
m
save/AssignVariableOp_169AssignVariableOpblock_11_expand_BN/moving_meansave/Identity_169*
dtype0
T
save/Identity_170Identitysave/RestoreV2:170*
T0*
_output_shapes
:
q
save/AssignVariableOp_170AssignVariableOp"block_11_expand_BN/moving_variancesave/Identity_170*
dtype0
T
save/Identity_171Identitysave/RestoreV2:171*
T0*
_output_shapes
:
r
save/AssignVariableOp_171AssignVariableOp#block_11_depthwise/depthwise_kernelsave/Identity_171*
dtype0
T
save/Identity_172Identitysave/RestoreV2:172*
T0*
_output_shapes
:
i
save/AssignVariableOp_172AssignVariableOpblock_11_depthwise_BN/betasave/Identity_172*
dtype0
T
save/Identity_173Identitysave/RestoreV2:173*
T0*
_output_shapes
:
j
save/AssignVariableOp_173AssignVariableOpblock_11_depthwise_BN/gammasave/Identity_173*
dtype0
T
save/Identity_174Identitysave/RestoreV2:174*
T0*
_output_shapes
:
p
save/AssignVariableOp_174AssignVariableOp!block_11_depthwise_BN/moving_meansave/Identity_174*
dtype0
T
save/Identity_175Identitysave/RestoreV2:175*
T0*
_output_shapes
:
t
save/AssignVariableOp_175AssignVariableOp%block_11_depthwise_BN/moving_variancesave/Identity_175*
dtype0
T
save/Identity_176Identitysave/RestoreV2:176*
T0*
_output_shapes
:
e
save/AssignVariableOp_176AssignVariableOpblock_1_expand_BN/betasave/Identity_176*
dtype0
T
save/Identity_177Identitysave/RestoreV2:177*
T0*
_output_shapes
:
f
save/AssignVariableOp_177AssignVariableOpblock_1_expand_BN/gammasave/Identity_177*
dtype0
T
save/Identity_178Identitysave/RestoreV2:178*
T0*
_output_shapes
:
l
save/AssignVariableOp_178AssignVariableOpblock_1_expand_BN/moving_meansave/Identity_178*
dtype0
T
save/Identity_179Identitysave/RestoreV2:179*
T0*
_output_shapes
:
p
save/AssignVariableOp_179AssignVariableOp!block_1_expand_BN/moving_variancesave/Identity_179*
dtype0
T
save/Identity_180Identitysave/RestoreV2:180*
T0*
_output_shapes
:
f
save/AssignVariableOp_180AssignVariableOpblock_11_project/kernelsave/Identity_180*
dtype0
T
save/Identity_181Identitysave/RestoreV2:181*
T0*
_output_shapes
:
g
save/AssignVariableOp_181AssignVariableOpblock_11_project_BN/betasave/Identity_181*
dtype0
T
save/Identity_182Identitysave/RestoreV2:182*
T0*
_output_shapes
:
h
save/AssignVariableOp_182AssignVariableOpblock_11_project_BN/gammasave/Identity_182*
dtype0
T
save/Identity_183Identitysave/RestoreV2:183*
T0*
_output_shapes
:
n
save/AssignVariableOp_183AssignVariableOpblock_11_project_BN/moving_meansave/Identity_183*
dtype0
T
save/Identity_184Identitysave/RestoreV2:184*
T0*
_output_shapes
:
r
save/AssignVariableOp_184AssignVariableOp#block_11_project_BN/moving_variancesave/Identity_184*
dtype0
T
save/Identity_185Identitysave/RestoreV2:185*
T0*
_output_shapes
:
e
save/AssignVariableOp_185AssignVariableOpblock_12_expand/kernelsave/Identity_185*
dtype0
T
save/Identity_186Identitysave/RestoreV2:186*
T0*
_output_shapes
:
f
save/AssignVariableOp_186AssignVariableOpblock_12_expand_BN/betasave/Identity_186*
dtype0
T
save/Identity_187Identitysave/RestoreV2:187*
T0*
_output_shapes
:
g
save/AssignVariableOp_187AssignVariableOpblock_12_expand_BN/gammasave/Identity_187*
dtype0
T
save/Identity_188Identitysave/RestoreV2:188*
T0*
_output_shapes
:
m
save/AssignVariableOp_188AssignVariableOpblock_12_expand_BN/moving_meansave/Identity_188*
dtype0
T
save/Identity_189Identitysave/RestoreV2:189*
T0*
_output_shapes
:
q
save/AssignVariableOp_189AssignVariableOp"block_12_expand_BN/moving_variancesave/Identity_189*
dtype0
T
save/Identity_190Identitysave/RestoreV2:190*
T0*
_output_shapes
:
r
save/AssignVariableOp_190AssignVariableOp#block_12_depthwise/depthwise_kernelsave/Identity_190*
dtype0
T
save/Identity_191Identitysave/RestoreV2:191*
T0*
_output_shapes
:
i
save/AssignVariableOp_191AssignVariableOpblock_12_depthwise_BN/betasave/Identity_191*
dtype0
T
save/Identity_192Identitysave/RestoreV2:192*
T0*
_output_shapes
:
j
save/AssignVariableOp_192AssignVariableOpblock_12_depthwise_BN/gammasave/Identity_192*
dtype0
T
save/Identity_193Identitysave/RestoreV2:193*
T0*
_output_shapes
:
p
save/AssignVariableOp_193AssignVariableOp!block_12_depthwise_BN/moving_meansave/Identity_193*
dtype0
T
save/Identity_194Identitysave/RestoreV2:194*
T0*
_output_shapes
:
t
save/AssignVariableOp_194AssignVariableOp%block_12_depthwise_BN/moving_variancesave/Identity_194*
dtype0
T
save/Identity_195Identitysave/RestoreV2:195*
T0*
_output_shapes
:
f
save/AssignVariableOp_195AssignVariableOpblock_12_project/kernelsave/Identity_195*
dtype0
T
save/Identity_196Identitysave/RestoreV2:196*
T0*
_output_shapes
:
g
save/AssignVariableOp_196AssignVariableOpblock_12_project_BN/betasave/Identity_196*
dtype0
T
save/Identity_197Identitysave/RestoreV2:197*
T0*
_output_shapes
:
h
save/AssignVariableOp_197AssignVariableOpblock_12_project_BN/gammasave/Identity_197*
dtype0
T
save/Identity_198Identitysave/RestoreV2:198*
T0*
_output_shapes
:
n
save/AssignVariableOp_198AssignVariableOpblock_12_project_BN/moving_meansave/Identity_198*
dtype0
T
save/Identity_199Identitysave/RestoreV2:199*
T0*
_output_shapes
:
r
save/AssignVariableOp_199AssignVariableOp#block_12_project_BN/moving_variancesave/Identity_199*
dtype0
T
save/Identity_200Identitysave/RestoreV2:200*
T0*
_output_shapes
:
e
save/AssignVariableOp_200AssignVariableOpblock_13_expand/kernelsave/Identity_200*
dtype0
T
save/Identity_201Identitysave/RestoreV2:201*
T0*
_output_shapes
:
f
save/AssignVariableOp_201AssignVariableOpblock_13_expand_BN/betasave/Identity_201*
dtype0
T
save/Identity_202Identitysave/RestoreV2:202*
T0*
_output_shapes
:
g
save/AssignVariableOp_202AssignVariableOpblock_13_expand_BN/gammasave/Identity_202*
dtype0
T
save/Identity_203Identitysave/RestoreV2:203*
T0*
_output_shapes
:
m
save/AssignVariableOp_203AssignVariableOpblock_13_expand_BN/moving_meansave/Identity_203*
dtype0
T
save/Identity_204Identitysave/RestoreV2:204*
T0*
_output_shapes
:
q
save/AssignVariableOp_204AssignVariableOp"block_13_expand_BN/moving_variancesave/Identity_204*
dtype0
T
save/Identity_205Identitysave/RestoreV2:205*
T0*
_output_shapes
:
q
save/AssignVariableOp_205AssignVariableOp"block_1_depthwise/depthwise_kernelsave/Identity_205*
dtype0
T
save/Identity_206Identitysave/RestoreV2:206*
T0*
_output_shapes
:
r
save/AssignVariableOp_206AssignVariableOp#block_13_depthwise/depthwise_kernelsave/Identity_206*
dtype0
T
save/Identity_207Identitysave/RestoreV2:207*
T0*
_output_shapes
:
i
save/AssignVariableOp_207AssignVariableOpblock_13_depthwise_BN/betasave/Identity_207*
dtype0
T
save/Identity_208Identitysave/RestoreV2:208*
T0*
_output_shapes
:
j
save/AssignVariableOp_208AssignVariableOpblock_13_depthwise_BN/gammasave/Identity_208*
dtype0
T
save/Identity_209Identitysave/RestoreV2:209*
T0*
_output_shapes
:
p
save/AssignVariableOp_209AssignVariableOp!block_13_depthwise_BN/moving_meansave/Identity_209*
dtype0
T
save/Identity_210Identitysave/RestoreV2:210*
T0*
_output_shapes
:
t
save/AssignVariableOp_210AssignVariableOp%block_13_depthwise_BN/moving_variancesave/Identity_210*
dtype0
T
save/Identity_211Identitysave/RestoreV2:211*
T0*
_output_shapes
:
f
save/AssignVariableOp_211AssignVariableOpblock_13_project/kernelsave/Identity_211*
dtype0
T
save/Identity_212Identitysave/RestoreV2:212*
T0*
_output_shapes
:
g
save/AssignVariableOp_212AssignVariableOpblock_13_project_BN/betasave/Identity_212*
dtype0
T
save/Identity_213Identitysave/RestoreV2:213*
T0*
_output_shapes
:
h
save/AssignVariableOp_213AssignVariableOpblock_13_project_BN/gammasave/Identity_213*
dtype0
T
save/Identity_214Identitysave/RestoreV2:214*
T0*
_output_shapes
:
n
save/AssignVariableOp_214AssignVariableOpblock_13_project_BN/moving_meansave/Identity_214*
dtype0
T
save/Identity_215Identitysave/RestoreV2:215*
T0*
_output_shapes
:
r
save/AssignVariableOp_215AssignVariableOp#block_13_project_BN/moving_variancesave/Identity_215*
dtype0
T
save/Identity_216Identitysave/RestoreV2:216*
T0*
_output_shapes
:
e
save/AssignVariableOp_216AssignVariableOpblock_14_expand/kernelsave/Identity_216*
dtype0
T
save/Identity_217Identitysave/RestoreV2:217*
T0*
_output_shapes
:
f
save/AssignVariableOp_217AssignVariableOpblock_14_expand_BN/betasave/Identity_217*
dtype0
T
save/Identity_218Identitysave/RestoreV2:218*
T0*
_output_shapes
:
g
save/AssignVariableOp_218AssignVariableOpblock_14_expand_BN/gammasave/Identity_218*
dtype0
T
save/Identity_219Identitysave/RestoreV2:219*
T0*
_output_shapes
:
m
save/AssignVariableOp_219AssignVariableOpblock_14_expand_BN/moving_meansave/Identity_219*
dtype0
T
save/Identity_220Identitysave/RestoreV2:220*
T0*
_output_shapes
:
q
save/AssignVariableOp_220AssignVariableOp"block_14_expand_BN/moving_variancesave/Identity_220*
dtype0
T
save/Identity_221Identitysave/RestoreV2:221*
T0*
_output_shapes
:
r
save/AssignVariableOp_221AssignVariableOp#block_14_depthwise/depthwise_kernelsave/Identity_221*
dtype0
T
save/Identity_222Identitysave/RestoreV2:222*
T0*
_output_shapes
:
i
save/AssignVariableOp_222AssignVariableOpblock_14_depthwise_BN/betasave/Identity_222*
dtype0
T
save/Identity_223Identitysave/RestoreV2:223*
T0*
_output_shapes
:
j
save/AssignVariableOp_223AssignVariableOpblock_14_depthwise_BN/gammasave/Identity_223*
dtype0
T
save/Identity_224Identitysave/RestoreV2:224*
T0*
_output_shapes
:
p
save/AssignVariableOp_224AssignVariableOp!block_14_depthwise_BN/moving_meansave/Identity_224*
dtype0
T
save/Identity_225Identitysave/RestoreV2:225*
T0*
_output_shapes
:
t
save/AssignVariableOp_225AssignVariableOp%block_14_depthwise_BN/moving_variancesave/Identity_225*
dtype0
T
save/Identity_226Identitysave/RestoreV2:226*
T0*
_output_shapes
:
f
save/AssignVariableOp_226AssignVariableOpblock_14_project/kernelsave/Identity_226*
dtype0
T
save/Identity_227Identitysave/RestoreV2:227*
T0*
_output_shapes
:
g
save/AssignVariableOp_227AssignVariableOpblock_14_project_BN/betasave/Identity_227*
dtype0
T
save/Identity_228Identitysave/RestoreV2:228*
T0*
_output_shapes
:
h
save/AssignVariableOp_228AssignVariableOpblock_14_project_BN/gammasave/Identity_228*
dtype0
T
save/Identity_229Identitysave/RestoreV2:229*
T0*
_output_shapes
:
n
save/AssignVariableOp_229AssignVariableOpblock_14_project_BN/moving_meansave/Identity_229*
dtype0
T
save/Identity_230Identitysave/RestoreV2:230*
T0*
_output_shapes
:
r
save/AssignVariableOp_230AssignVariableOp#block_14_project_BN/moving_variancesave/Identity_230*
dtype0
T
save/Identity_231Identitysave/RestoreV2:231*
T0*
_output_shapes
:
h
save/AssignVariableOp_231AssignVariableOpblock_1_depthwise_BN/betasave/Identity_231*
dtype0
T
save/Identity_232Identitysave/RestoreV2:232*
T0*
_output_shapes
:
i
save/AssignVariableOp_232AssignVariableOpblock_1_depthwise_BN/gammasave/Identity_232*
dtype0
T
save/Identity_233Identitysave/RestoreV2:233*
T0*
_output_shapes
:
o
save/AssignVariableOp_233AssignVariableOp block_1_depthwise_BN/moving_meansave/Identity_233*
dtype0
T
save/Identity_234Identitysave/RestoreV2:234*
T0*
_output_shapes
:
s
save/AssignVariableOp_234AssignVariableOp$block_1_depthwise_BN/moving_variancesave/Identity_234*
dtype0
T
save/Identity_235Identitysave/RestoreV2:235*
T0*
_output_shapes
:
e
save/AssignVariableOp_235AssignVariableOpblock_15_expand/kernelsave/Identity_235*
dtype0
T
save/Identity_236Identitysave/RestoreV2:236*
T0*
_output_shapes
:
f
save/AssignVariableOp_236AssignVariableOpblock_15_expand_BN/betasave/Identity_236*
dtype0
T
save/Identity_237Identitysave/RestoreV2:237*
T0*
_output_shapes
:
g
save/AssignVariableOp_237AssignVariableOpblock_15_expand_BN/gammasave/Identity_237*
dtype0
T
save/Identity_238Identitysave/RestoreV2:238*
T0*
_output_shapes
:
m
save/AssignVariableOp_238AssignVariableOpblock_15_expand_BN/moving_meansave/Identity_238*
dtype0
T
save/Identity_239Identitysave/RestoreV2:239*
T0*
_output_shapes
:
q
save/AssignVariableOp_239AssignVariableOp"block_15_expand_BN/moving_variancesave/Identity_239*
dtype0
T
save/Identity_240Identitysave/RestoreV2:240*
T0*
_output_shapes
:
r
save/AssignVariableOp_240AssignVariableOp#block_15_depthwise/depthwise_kernelsave/Identity_240*
dtype0
T
save/Identity_241Identitysave/RestoreV2:241*
T0*
_output_shapes
:
i
save/AssignVariableOp_241AssignVariableOpblock_15_depthwise_BN/betasave/Identity_241*
dtype0
T
save/Identity_242Identitysave/RestoreV2:242*
T0*
_output_shapes
:
j
save/AssignVariableOp_242AssignVariableOpblock_15_depthwise_BN/gammasave/Identity_242*
dtype0
T
save/Identity_243Identitysave/RestoreV2:243*
T0*
_output_shapes
:
p
save/AssignVariableOp_243AssignVariableOp!block_15_depthwise_BN/moving_meansave/Identity_243*
dtype0
T
save/Identity_244Identitysave/RestoreV2:244*
T0*
_output_shapes
:
t
save/AssignVariableOp_244AssignVariableOp%block_15_depthwise_BN/moving_variancesave/Identity_244*
dtype0
T
save/Identity_245Identitysave/RestoreV2:245*
T0*
_output_shapes
:
f
save/AssignVariableOp_245AssignVariableOpblock_15_project/kernelsave/Identity_245*
dtype0
T
save/Identity_246Identitysave/RestoreV2:246*
T0*
_output_shapes
:
g
save/AssignVariableOp_246AssignVariableOpblock_15_project_BN/betasave/Identity_246*
dtype0
T
save/Identity_247Identitysave/RestoreV2:247*
T0*
_output_shapes
:
h
save/AssignVariableOp_247AssignVariableOpblock_15_project_BN/gammasave/Identity_247*
dtype0
T
save/Identity_248Identitysave/RestoreV2:248*
T0*
_output_shapes
:
n
save/AssignVariableOp_248AssignVariableOpblock_15_project_BN/moving_meansave/Identity_248*
dtype0
T
save/Identity_249Identitysave/RestoreV2:249*
T0*
_output_shapes
:
r
save/AssignVariableOp_249AssignVariableOp#block_15_project_BN/moving_variancesave/Identity_249*
dtype0
T
save/Identity_250Identitysave/RestoreV2:250*
T0*
_output_shapes
:
e
save/AssignVariableOp_250AssignVariableOpblock_16_expand/kernelsave/Identity_250*
dtype0
T
save/Identity_251Identitysave/RestoreV2:251*
T0*
_output_shapes
:
f
save/AssignVariableOp_251AssignVariableOpblock_16_expand_BN/betasave/Identity_251*
dtype0
T
save/Identity_252Identitysave/RestoreV2:252*
T0*
_output_shapes
:
g
save/AssignVariableOp_252AssignVariableOpblock_16_expand_BN/gammasave/Identity_252*
dtype0
T
save/Identity_253Identitysave/RestoreV2:253*
T0*
_output_shapes
:
m
save/AssignVariableOp_253AssignVariableOpblock_16_expand_BN/moving_meansave/Identity_253*
dtype0
T
save/Identity_254Identitysave/RestoreV2:254*
T0*
_output_shapes
:
q
save/AssignVariableOp_254AssignVariableOp"block_16_expand_BN/moving_variancesave/Identity_254*
dtype0
T
save/Identity_255Identitysave/RestoreV2:255*
T0*
_output_shapes
:
r
save/AssignVariableOp_255AssignVariableOp#block_16_depthwise/depthwise_kernelsave/Identity_255*
dtype0
T
save/Identity_256Identitysave/RestoreV2:256*
T0*
_output_shapes
:
i
save/AssignVariableOp_256AssignVariableOpblock_16_depthwise_BN/betasave/Identity_256*
dtype0
T
save/Identity_257Identitysave/RestoreV2:257*
T0*
_output_shapes
:
j
save/AssignVariableOp_257AssignVariableOpblock_16_depthwise_BN/gammasave/Identity_257*
dtype0
T
save/Identity_258Identitysave/RestoreV2:258*
T0*
_output_shapes
:
p
save/AssignVariableOp_258AssignVariableOp!block_16_depthwise_BN/moving_meansave/Identity_258*
dtype0
T
save/Identity_259Identitysave/RestoreV2:259*
T0*
_output_shapes
:
t
save/AssignVariableOp_259AssignVariableOp%block_16_depthwise_BN/moving_variancesave/Identity_259*
dtype0
�8
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_100^save/AssignVariableOp_101^save/AssignVariableOp_102^save/AssignVariableOp_103^save/AssignVariableOp_104^save/AssignVariableOp_105^save/AssignVariableOp_106^save/AssignVariableOp_107^save/AssignVariableOp_108^save/AssignVariableOp_109^save/AssignVariableOp_11^save/AssignVariableOp_110^save/AssignVariableOp_111^save/AssignVariableOp_112^save/AssignVariableOp_113^save/AssignVariableOp_114^save/AssignVariableOp_115^save/AssignVariableOp_116^save/AssignVariableOp_117^save/AssignVariableOp_118^save/AssignVariableOp_119^save/AssignVariableOp_12^save/AssignVariableOp_120^save/AssignVariableOp_121^save/AssignVariableOp_122^save/AssignVariableOp_123^save/AssignVariableOp_124^save/AssignVariableOp_125^save/AssignVariableOp_126^save/AssignVariableOp_127^save/AssignVariableOp_128^save/AssignVariableOp_129^save/AssignVariableOp_13^save/AssignVariableOp_130^save/AssignVariableOp_131^save/AssignVariableOp_132^save/AssignVariableOp_133^save/AssignVariableOp_134^save/AssignVariableOp_135^save/AssignVariableOp_136^save/AssignVariableOp_137^save/AssignVariableOp_138^save/AssignVariableOp_139^save/AssignVariableOp_14^save/AssignVariableOp_140^save/AssignVariableOp_141^save/AssignVariableOp_142^save/AssignVariableOp_143^save/AssignVariableOp_144^save/AssignVariableOp_145^save/AssignVariableOp_146^save/AssignVariableOp_147^save/AssignVariableOp_148^save/AssignVariableOp_149^save/AssignVariableOp_15^save/AssignVariableOp_150^save/AssignVariableOp_151^save/AssignVariableOp_152^save/AssignVariableOp_153^save/AssignVariableOp_154^save/AssignVariableOp_155^save/AssignVariableOp_156^save/AssignVariableOp_157^save/AssignVariableOp_158^save/AssignVariableOp_159^save/AssignVariableOp_16^save/AssignVariableOp_160^save/AssignVariableOp_161^save/AssignVariableOp_162^save/AssignVariableOp_163^save/AssignVariableOp_164^save/AssignVariableOp_165^save/AssignVariableOp_166^save/AssignVariableOp_167^save/AssignVariableOp_168^save/AssignVariableOp_169^save/AssignVariableOp_17^save/AssignVariableOp_170^save/AssignVariableOp_171^save/AssignVariableOp_172^save/AssignVariableOp_173^save/AssignVariableOp_174^save/AssignVariableOp_175^save/AssignVariableOp_176^save/AssignVariableOp_177^save/AssignVariableOp_178^save/AssignVariableOp_179^save/AssignVariableOp_18^save/AssignVariableOp_180^save/AssignVariableOp_181^save/AssignVariableOp_182^save/AssignVariableOp_183^save/AssignVariableOp_184^save/AssignVariableOp_185^save/AssignVariableOp_186^save/AssignVariableOp_187^save/AssignVariableOp_188^save/AssignVariableOp_189^save/AssignVariableOp_19^save/AssignVariableOp_190^save/AssignVariableOp_191^save/AssignVariableOp_192^save/AssignVariableOp_193^save/AssignVariableOp_194^save/AssignVariableOp_195^save/AssignVariableOp_196^save/AssignVariableOp_197^save/AssignVariableOp_198^save/AssignVariableOp_199^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_200^save/AssignVariableOp_201^save/AssignVariableOp_202^save/AssignVariableOp_203^save/AssignVariableOp_204^save/AssignVariableOp_205^save/AssignVariableOp_206^save/AssignVariableOp_207^save/AssignVariableOp_208^save/AssignVariableOp_209^save/AssignVariableOp_21^save/AssignVariableOp_210^save/AssignVariableOp_211^save/AssignVariableOp_212^save/AssignVariableOp_213^save/AssignVariableOp_214^save/AssignVariableOp_215^save/AssignVariableOp_216^save/AssignVariableOp_217^save/AssignVariableOp_218^save/AssignVariableOp_219^save/AssignVariableOp_22^save/AssignVariableOp_220^save/AssignVariableOp_221^save/AssignVariableOp_222^save/AssignVariableOp_223^save/AssignVariableOp_224^save/AssignVariableOp_225^save/AssignVariableOp_226^save/AssignVariableOp_227^save/AssignVariableOp_228^save/AssignVariableOp_229^save/AssignVariableOp_23^save/AssignVariableOp_230^save/AssignVariableOp_231^save/AssignVariableOp_232^save/AssignVariableOp_233^save/AssignVariableOp_234^save/AssignVariableOp_235^save/AssignVariableOp_236^save/AssignVariableOp_237^save/AssignVariableOp_238^save/AssignVariableOp_239^save/AssignVariableOp_24^save/AssignVariableOp_240^save/AssignVariableOp_241^save/AssignVariableOp_242^save/AssignVariableOp_243^save/AssignVariableOp_244^save/AssignVariableOp_245^save/AssignVariableOp_246^save/AssignVariableOp_247^save/AssignVariableOp_248^save/AssignVariableOp_249^save/AssignVariableOp_25^save/AssignVariableOp_250^save/AssignVariableOp_251^save/AssignVariableOp_252^save/AssignVariableOp_253^save/AssignVariableOp_254^save/AssignVariableOp_255^save/AssignVariableOp_256^save/AssignVariableOp_257^save/AssignVariableOp_258^save/AssignVariableOp_259^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_39^save/AssignVariableOp_4^save/AssignVariableOp_40^save/AssignVariableOp_41^save/AssignVariableOp_42^save/AssignVariableOp_43^save/AssignVariableOp_44^save/AssignVariableOp_45^save/AssignVariableOp_46^save/AssignVariableOp_47^save/AssignVariableOp_48^save/AssignVariableOp_49^save/AssignVariableOp_5^save/AssignVariableOp_50^save/AssignVariableOp_51^save/AssignVariableOp_52^save/AssignVariableOp_53^save/AssignVariableOp_54^save/AssignVariableOp_55^save/AssignVariableOp_56^save/AssignVariableOp_57^save/AssignVariableOp_58^save/AssignVariableOp_59^save/AssignVariableOp_6^save/AssignVariableOp_60^save/AssignVariableOp_61^save/AssignVariableOp_62^save/AssignVariableOp_63^save/AssignVariableOp_64^save/AssignVariableOp_65^save/AssignVariableOp_66^save/AssignVariableOp_67^save/AssignVariableOp_68^save/AssignVariableOp_69^save/AssignVariableOp_7^save/AssignVariableOp_70^save/AssignVariableOp_71^save/AssignVariableOp_72^save/AssignVariableOp_73^save/AssignVariableOp_74^save/AssignVariableOp_75^save/AssignVariableOp_76^save/AssignVariableOp_77^save/AssignVariableOp_78^save/AssignVariableOp_79^save/AssignVariableOp_8^save/AssignVariableOp_80^save/AssignVariableOp_81^save/AssignVariableOp_82^save/AssignVariableOp_83^save/AssignVariableOp_84^save/AssignVariableOp_85^save/AssignVariableOp_86^save/AssignVariableOp_87^save/AssignVariableOp_88^save/AssignVariableOp_89^save/AssignVariableOp_9^save/AssignVariableOp_90^save/AssignVariableOp_91^save/AssignVariableOp_92^save/AssignVariableOp_93^save/AssignVariableOp_94^save/AssignVariableOp_95^save/AssignVariableOp_96^save/AssignVariableOp_97^save/AssignVariableOp_98^save/AssignVariableOp_99

init_1NoOp"&D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"��
trainable_variables����
x
Conv1/kernel:0Conv1/kernel/Assign"Conv1/kernel/Read/ReadVariableOp:0(2)Conv1/kernel/Initializer/random_uniform:08
v
bn_Conv1/gamma:0bn_Conv1/gamma/Assign$bn_Conv1/gamma/Read/ReadVariableOp:0(2!bn_Conv1/gamma/Initializer/ones:08
s
bn_Conv1/beta:0bn_Conv1/beta/Assign#bn_Conv1/beta/Read/ReadVariableOp:0(2!bn_Conv1/beta/Initializer/zeros:08
�
*expanded_conv_depthwise/depthwise_kernel:0/expanded_conv_depthwise/depthwise_kernel/Assign>expanded_conv_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2Eexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
"expanded_conv_depthwise_BN/gamma:0'expanded_conv_depthwise_BN/gamma/Assign6expanded_conv_depthwise_BN/gamma/Read/ReadVariableOp:0(23expanded_conv_depthwise_BN/gamma/Initializer/ones:08
�
!expanded_conv_depthwise_BN/beta:0&expanded_conv_depthwise_BN/beta/Assign5expanded_conv_depthwise_BN/beta/Read/ReadVariableOp:0(23expanded_conv_depthwise_BN/beta/Initializer/zeros:08
�
expanded_conv_project/kernel:0#expanded_conv_project/kernel/Assign2expanded_conv_project/kernel/Read/ReadVariableOp:0(29expanded_conv_project/kernel/Initializer/random_uniform:08
�
 expanded_conv_project_BN/gamma:0%expanded_conv_project_BN/gamma/Assign4expanded_conv_project_BN/gamma/Read/ReadVariableOp:0(21expanded_conv_project_BN/gamma/Initializer/ones:08
�
expanded_conv_project_BN/beta:0$expanded_conv_project_BN/beta/Assign3expanded_conv_project_BN/beta/Read/ReadVariableOp:0(21expanded_conv_project_BN/beta/Initializer/zeros:08
�
block_1_expand/kernel:0block_1_expand/kernel/Assign+block_1_expand/kernel/Read/ReadVariableOp:0(22block_1_expand/kernel/Initializer/random_uniform:08
�
block_1_expand_BN/gamma:0block_1_expand_BN/gamma/Assign-block_1_expand_BN/gamma/Read/ReadVariableOp:0(2*block_1_expand_BN/gamma/Initializer/ones:08
�
block_1_expand_BN/beta:0block_1_expand_BN/beta/Assign,block_1_expand_BN/beta/Read/ReadVariableOp:0(2*block_1_expand_BN/beta/Initializer/zeros:08
�
$block_1_depthwise/depthwise_kernel:0)block_1_depthwise/depthwise_kernel/Assign8block_1_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_1_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_1_depthwise_BN/gamma:0!block_1_depthwise_BN/gamma/Assign0block_1_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_1_depthwise_BN/gamma/Initializer/ones:08
�
block_1_depthwise_BN/beta:0 block_1_depthwise_BN/beta/Assign/block_1_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_1_depthwise_BN/beta/Initializer/zeros:08
�
block_1_project/kernel:0block_1_project/kernel/Assign,block_1_project/kernel/Read/ReadVariableOp:0(23block_1_project/kernel/Initializer/random_uniform:08
�
block_1_project_BN/gamma:0block_1_project_BN/gamma/Assign.block_1_project_BN/gamma/Read/ReadVariableOp:0(2+block_1_project_BN/gamma/Initializer/ones:08
�
block_1_project_BN/beta:0block_1_project_BN/beta/Assign-block_1_project_BN/beta/Read/ReadVariableOp:0(2+block_1_project_BN/beta/Initializer/zeros:08
�
block_2_expand/kernel:0block_2_expand/kernel/Assign+block_2_expand/kernel/Read/ReadVariableOp:0(22block_2_expand/kernel/Initializer/random_uniform:08
�
block_2_expand_BN/gamma:0block_2_expand_BN/gamma/Assign-block_2_expand_BN/gamma/Read/ReadVariableOp:0(2*block_2_expand_BN/gamma/Initializer/ones:08
�
block_2_expand_BN/beta:0block_2_expand_BN/beta/Assign,block_2_expand_BN/beta/Read/ReadVariableOp:0(2*block_2_expand_BN/beta/Initializer/zeros:08
�
$block_2_depthwise/depthwise_kernel:0)block_2_depthwise/depthwise_kernel/Assign8block_2_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_2_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_2_depthwise_BN/gamma:0!block_2_depthwise_BN/gamma/Assign0block_2_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_2_depthwise_BN/gamma/Initializer/ones:08
�
block_2_depthwise_BN/beta:0 block_2_depthwise_BN/beta/Assign/block_2_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_2_depthwise_BN/beta/Initializer/zeros:08
�
block_2_project/kernel:0block_2_project/kernel/Assign,block_2_project/kernel/Read/ReadVariableOp:0(23block_2_project/kernel/Initializer/random_uniform:08
�
block_2_project_BN/gamma:0block_2_project_BN/gamma/Assign.block_2_project_BN/gamma/Read/ReadVariableOp:0(2+block_2_project_BN/gamma/Initializer/ones:08
�
block_2_project_BN/beta:0block_2_project_BN/beta/Assign-block_2_project_BN/beta/Read/ReadVariableOp:0(2+block_2_project_BN/beta/Initializer/zeros:08
�
block_3_expand/kernel:0block_3_expand/kernel/Assign+block_3_expand/kernel/Read/ReadVariableOp:0(22block_3_expand/kernel/Initializer/random_uniform:08
�
block_3_expand_BN/gamma:0block_3_expand_BN/gamma/Assign-block_3_expand_BN/gamma/Read/ReadVariableOp:0(2*block_3_expand_BN/gamma/Initializer/ones:08
�
block_3_expand_BN/beta:0block_3_expand_BN/beta/Assign,block_3_expand_BN/beta/Read/ReadVariableOp:0(2*block_3_expand_BN/beta/Initializer/zeros:08
�
$block_3_depthwise/depthwise_kernel:0)block_3_depthwise/depthwise_kernel/Assign8block_3_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_3_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_3_depthwise_BN/gamma:0!block_3_depthwise_BN/gamma/Assign0block_3_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_3_depthwise_BN/gamma/Initializer/ones:08
�
block_3_depthwise_BN/beta:0 block_3_depthwise_BN/beta/Assign/block_3_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_3_depthwise_BN/beta/Initializer/zeros:08
�
block_3_project/kernel:0block_3_project/kernel/Assign,block_3_project/kernel/Read/ReadVariableOp:0(23block_3_project/kernel/Initializer/random_uniform:08
�
block_3_project_BN/gamma:0block_3_project_BN/gamma/Assign.block_3_project_BN/gamma/Read/ReadVariableOp:0(2+block_3_project_BN/gamma/Initializer/ones:08
�
block_3_project_BN/beta:0block_3_project_BN/beta/Assign-block_3_project_BN/beta/Read/ReadVariableOp:0(2+block_3_project_BN/beta/Initializer/zeros:08
�
block_4_expand/kernel:0block_4_expand/kernel/Assign+block_4_expand/kernel/Read/ReadVariableOp:0(22block_4_expand/kernel/Initializer/random_uniform:08
�
block_4_expand_BN/gamma:0block_4_expand_BN/gamma/Assign-block_4_expand_BN/gamma/Read/ReadVariableOp:0(2*block_4_expand_BN/gamma/Initializer/ones:08
�
block_4_expand_BN/beta:0block_4_expand_BN/beta/Assign,block_4_expand_BN/beta/Read/ReadVariableOp:0(2*block_4_expand_BN/beta/Initializer/zeros:08
�
$block_4_depthwise/depthwise_kernel:0)block_4_depthwise/depthwise_kernel/Assign8block_4_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_4_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_4_depthwise_BN/gamma:0!block_4_depthwise_BN/gamma/Assign0block_4_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_4_depthwise_BN/gamma/Initializer/ones:08
�
block_4_depthwise_BN/beta:0 block_4_depthwise_BN/beta/Assign/block_4_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_4_depthwise_BN/beta/Initializer/zeros:08
�
block_4_project/kernel:0block_4_project/kernel/Assign,block_4_project/kernel/Read/ReadVariableOp:0(23block_4_project/kernel/Initializer/random_uniform:08
�
block_4_project_BN/gamma:0block_4_project_BN/gamma/Assign.block_4_project_BN/gamma/Read/ReadVariableOp:0(2+block_4_project_BN/gamma/Initializer/ones:08
�
block_4_project_BN/beta:0block_4_project_BN/beta/Assign-block_4_project_BN/beta/Read/ReadVariableOp:0(2+block_4_project_BN/beta/Initializer/zeros:08
�
block_5_expand/kernel:0block_5_expand/kernel/Assign+block_5_expand/kernel/Read/ReadVariableOp:0(22block_5_expand/kernel/Initializer/random_uniform:08
�
block_5_expand_BN/gamma:0block_5_expand_BN/gamma/Assign-block_5_expand_BN/gamma/Read/ReadVariableOp:0(2*block_5_expand_BN/gamma/Initializer/ones:08
�
block_5_expand_BN/beta:0block_5_expand_BN/beta/Assign,block_5_expand_BN/beta/Read/ReadVariableOp:0(2*block_5_expand_BN/beta/Initializer/zeros:08
�
$block_5_depthwise/depthwise_kernel:0)block_5_depthwise/depthwise_kernel/Assign8block_5_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_5_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_5_depthwise_BN/gamma:0!block_5_depthwise_BN/gamma/Assign0block_5_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_5_depthwise_BN/gamma/Initializer/ones:08
�
block_5_depthwise_BN/beta:0 block_5_depthwise_BN/beta/Assign/block_5_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_5_depthwise_BN/beta/Initializer/zeros:08
�
block_5_project/kernel:0block_5_project/kernel/Assign,block_5_project/kernel/Read/ReadVariableOp:0(23block_5_project/kernel/Initializer/random_uniform:08
�
block_5_project_BN/gamma:0block_5_project_BN/gamma/Assign.block_5_project_BN/gamma/Read/ReadVariableOp:0(2+block_5_project_BN/gamma/Initializer/ones:08
�
block_5_project_BN/beta:0block_5_project_BN/beta/Assign-block_5_project_BN/beta/Read/ReadVariableOp:0(2+block_5_project_BN/beta/Initializer/zeros:08
�
block_6_expand/kernel:0block_6_expand/kernel/Assign+block_6_expand/kernel/Read/ReadVariableOp:0(22block_6_expand/kernel/Initializer/random_uniform:08
�
block_6_expand_BN/gamma:0block_6_expand_BN/gamma/Assign-block_6_expand_BN/gamma/Read/ReadVariableOp:0(2*block_6_expand_BN/gamma/Initializer/ones:08
�
block_6_expand_BN/beta:0block_6_expand_BN/beta/Assign,block_6_expand_BN/beta/Read/ReadVariableOp:0(2*block_6_expand_BN/beta/Initializer/zeros:08
�
$block_6_depthwise/depthwise_kernel:0)block_6_depthwise/depthwise_kernel/Assign8block_6_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_6_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_6_depthwise_BN/gamma:0!block_6_depthwise_BN/gamma/Assign0block_6_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_6_depthwise_BN/gamma/Initializer/ones:08
�
block_6_depthwise_BN/beta:0 block_6_depthwise_BN/beta/Assign/block_6_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_6_depthwise_BN/beta/Initializer/zeros:08
�
block_6_project/kernel:0block_6_project/kernel/Assign,block_6_project/kernel/Read/ReadVariableOp:0(23block_6_project/kernel/Initializer/random_uniform:08
�
block_6_project_BN/gamma:0block_6_project_BN/gamma/Assign.block_6_project_BN/gamma/Read/ReadVariableOp:0(2+block_6_project_BN/gamma/Initializer/ones:08
�
block_6_project_BN/beta:0block_6_project_BN/beta/Assign-block_6_project_BN/beta/Read/ReadVariableOp:0(2+block_6_project_BN/beta/Initializer/zeros:08
�
block_7_expand/kernel:0block_7_expand/kernel/Assign+block_7_expand/kernel/Read/ReadVariableOp:0(22block_7_expand/kernel/Initializer/random_uniform:08
�
block_7_expand_BN/gamma:0block_7_expand_BN/gamma/Assign-block_7_expand_BN/gamma/Read/ReadVariableOp:0(2*block_7_expand_BN/gamma/Initializer/ones:08
�
block_7_expand_BN/beta:0block_7_expand_BN/beta/Assign,block_7_expand_BN/beta/Read/ReadVariableOp:0(2*block_7_expand_BN/beta/Initializer/zeros:08
�
$block_7_depthwise/depthwise_kernel:0)block_7_depthwise/depthwise_kernel/Assign8block_7_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_7_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_7_depthwise_BN/gamma:0!block_7_depthwise_BN/gamma/Assign0block_7_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_7_depthwise_BN/gamma/Initializer/ones:08
�
block_7_depthwise_BN/beta:0 block_7_depthwise_BN/beta/Assign/block_7_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_7_depthwise_BN/beta/Initializer/zeros:08
�
block_7_project/kernel:0block_7_project/kernel/Assign,block_7_project/kernel/Read/ReadVariableOp:0(23block_7_project/kernel/Initializer/random_uniform:08
�
block_7_project_BN/gamma:0block_7_project_BN/gamma/Assign.block_7_project_BN/gamma/Read/ReadVariableOp:0(2+block_7_project_BN/gamma/Initializer/ones:08
�
block_7_project_BN/beta:0block_7_project_BN/beta/Assign-block_7_project_BN/beta/Read/ReadVariableOp:0(2+block_7_project_BN/beta/Initializer/zeros:08
�
block_8_expand/kernel:0block_8_expand/kernel/Assign+block_8_expand/kernel/Read/ReadVariableOp:0(22block_8_expand/kernel/Initializer/random_uniform:08
�
block_8_expand_BN/gamma:0block_8_expand_BN/gamma/Assign-block_8_expand_BN/gamma/Read/ReadVariableOp:0(2*block_8_expand_BN/gamma/Initializer/ones:08
�
block_8_expand_BN/beta:0block_8_expand_BN/beta/Assign,block_8_expand_BN/beta/Read/ReadVariableOp:0(2*block_8_expand_BN/beta/Initializer/zeros:08
�
$block_8_depthwise/depthwise_kernel:0)block_8_depthwise/depthwise_kernel/Assign8block_8_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_8_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_8_depthwise_BN/gamma:0!block_8_depthwise_BN/gamma/Assign0block_8_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_8_depthwise_BN/gamma/Initializer/ones:08
�
block_8_depthwise_BN/beta:0 block_8_depthwise_BN/beta/Assign/block_8_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_8_depthwise_BN/beta/Initializer/zeros:08
�
block_8_project/kernel:0block_8_project/kernel/Assign,block_8_project/kernel/Read/ReadVariableOp:0(23block_8_project/kernel/Initializer/random_uniform:08
�
block_8_project_BN/gamma:0block_8_project_BN/gamma/Assign.block_8_project_BN/gamma/Read/ReadVariableOp:0(2+block_8_project_BN/gamma/Initializer/ones:08
�
block_8_project_BN/beta:0block_8_project_BN/beta/Assign-block_8_project_BN/beta/Read/ReadVariableOp:0(2+block_8_project_BN/beta/Initializer/zeros:08
�
block_9_expand/kernel:0block_9_expand/kernel/Assign+block_9_expand/kernel/Read/ReadVariableOp:0(22block_9_expand/kernel/Initializer/random_uniform:08
�
block_9_expand_BN/gamma:0block_9_expand_BN/gamma/Assign-block_9_expand_BN/gamma/Read/ReadVariableOp:0(2*block_9_expand_BN/gamma/Initializer/ones:08
�
block_9_expand_BN/beta:0block_9_expand_BN/beta/Assign,block_9_expand_BN/beta/Read/ReadVariableOp:0(2*block_9_expand_BN/beta/Initializer/zeros:08
�
$block_9_depthwise/depthwise_kernel:0)block_9_depthwise/depthwise_kernel/Assign8block_9_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_9_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_9_depthwise_BN/gamma:0!block_9_depthwise_BN/gamma/Assign0block_9_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_9_depthwise_BN/gamma/Initializer/ones:08
�
block_9_depthwise_BN/beta:0 block_9_depthwise_BN/beta/Assign/block_9_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_9_depthwise_BN/beta/Initializer/zeros:08
�
block_9_project/kernel:0block_9_project/kernel/Assign,block_9_project/kernel/Read/ReadVariableOp:0(23block_9_project/kernel/Initializer/random_uniform:08
�
block_9_project_BN/gamma:0block_9_project_BN/gamma/Assign.block_9_project_BN/gamma/Read/ReadVariableOp:0(2+block_9_project_BN/gamma/Initializer/ones:08
�
block_9_project_BN/beta:0block_9_project_BN/beta/Assign-block_9_project_BN/beta/Read/ReadVariableOp:0(2+block_9_project_BN/beta/Initializer/zeros:08
�
block_10_expand/kernel:0block_10_expand/kernel/Assign,block_10_expand/kernel/Read/ReadVariableOp:0(23block_10_expand/kernel/Initializer/random_uniform:08
�
block_10_expand_BN/gamma:0block_10_expand_BN/gamma/Assign.block_10_expand_BN/gamma/Read/ReadVariableOp:0(2+block_10_expand_BN/gamma/Initializer/ones:08
�
block_10_expand_BN/beta:0block_10_expand_BN/beta/Assign-block_10_expand_BN/beta/Read/ReadVariableOp:0(2+block_10_expand_BN/beta/Initializer/zeros:08
�
%block_10_depthwise/depthwise_kernel:0*block_10_depthwise/depthwise_kernel/Assign9block_10_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2@block_10_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_10_depthwise_BN/gamma:0"block_10_depthwise_BN/gamma/Assign1block_10_depthwise_BN/gamma/Read/ReadVariableOp:0(2.block_10_depthwise_BN/gamma/Initializer/ones:08
�
block_10_depthwise_BN/beta:0!block_10_depthwise_BN/beta/Assign0block_10_depthwise_BN/beta/Read/ReadVariableOp:0(2.block_10_depthwise_BN/beta/Initializer/zeros:08
�
block_10_project/kernel:0block_10_project/kernel/Assign-block_10_project/kernel/Read/ReadVariableOp:0(24block_10_project/kernel/Initializer/random_uniform:08
�
block_10_project_BN/gamma:0 block_10_project_BN/gamma/Assign/block_10_project_BN/gamma/Read/ReadVariableOp:0(2,block_10_project_BN/gamma/Initializer/ones:08
�
block_10_project_BN/beta:0block_10_project_BN/beta/Assign.block_10_project_BN/beta/Read/ReadVariableOp:0(2,block_10_project_BN/beta/Initializer/zeros:08
�
block_11_expand/kernel:0block_11_expand/kernel/Assign,block_11_expand/kernel/Read/ReadVariableOp:0(23block_11_expand/kernel/Initializer/random_uniform:08
�
block_11_expand_BN/gamma:0block_11_expand_BN/gamma/Assign.block_11_expand_BN/gamma/Read/ReadVariableOp:0(2+block_11_expand_BN/gamma/Initializer/ones:08
�
block_11_expand_BN/beta:0block_11_expand_BN/beta/Assign-block_11_expand_BN/beta/Read/ReadVariableOp:0(2+block_11_expand_BN/beta/Initializer/zeros:08
�
%block_11_depthwise/depthwise_kernel:0*block_11_depthwise/depthwise_kernel/Assign9block_11_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2@block_11_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_11_depthwise_BN/gamma:0"block_11_depthwise_BN/gamma/Assign1block_11_depthwise_BN/gamma/Read/ReadVariableOp:0(2.block_11_depthwise_BN/gamma/Initializer/ones:08
�
block_11_depthwise_BN/beta:0!block_11_depthwise_BN/beta/Assign0block_11_depthwise_BN/beta/Read/ReadVariableOp:0(2.block_11_depthwise_BN/beta/Initializer/zeros:08
�
block_11_project/kernel:0block_11_project/kernel/Assign-block_11_project/kernel/Read/ReadVariableOp:0(24block_11_project/kernel/Initializer/random_uniform:08
�
block_11_project_BN/gamma:0 block_11_project_BN/gamma/Assign/block_11_project_BN/gamma/Read/ReadVariableOp:0(2,block_11_project_BN/gamma/Initializer/ones:08
�
block_11_project_BN/beta:0block_11_project_BN/beta/Assign.block_11_project_BN/beta/Read/ReadVariableOp:0(2,block_11_project_BN/beta/Initializer/zeros:08
�
block_12_expand/kernel:0block_12_expand/kernel/Assign,block_12_expand/kernel/Read/ReadVariableOp:0(23block_12_expand/kernel/Initializer/random_uniform:08
�
block_12_expand_BN/gamma:0block_12_expand_BN/gamma/Assign.block_12_expand_BN/gamma/Read/ReadVariableOp:0(2+block_12_expand_BN/gamma/Initializer/ones:08
�
block_12_expand_BN/beta:0block_12_expand_BN/beta/Assign-block_12_expand_BN/beta/Read/ReadVariableOp:0(2+block_12_expand_BN/beta/Initializer/zeros:08
�
%block_12_depthwise/depthwise_kernel:0*block_12_depthwise/depthwise_kernel/Assign9block_12_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2@block_12_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_12_depthwise_BN/gamma:0"block_12_depthwise_BN/gamma/Assign1block_12_depthwise_BN/gamma/Read/ReadVariableOp:0(2.block_12_depthwise_BN/gamma/Initializer/ones:08
�
block_12_depthwise_BN/beta:0!block_12_depthwise_BN/beta/Assign0block_12_depthwise_BN/beta/Read/ReadVariableOp:0(2.block_12_depthwise_BN/beta/Initializer/zeros:08
�
block_12_project/kernel:0block_12_project/kernel/Assign-block_12_project/kernel/Read/ReadVariableOp:0(24block_12_project/kernel/Initializer/random_uniform:08
�
block_12_project_BN/gamma:0 block_12_project_BN/gamma/Assign/block_12_project_BN/gamma/Read/ReadVariableOp:0(2,block_12_project_BN/gamma/Initializer/ones:08
�
block_12_project_BN/beta:0block_12_project_BN/beta/Assign.block_12_project_BN/beta/Read/ReadVariableOp:0(2,block_12_project_BN/beta/Initializer/zeros:08
�
block_13_expand/kernel:0block_13_expand/kernel/Assign,block_13_expand/kernel/Read/ReadVariableOp:0(23block_13_expand/kernel/Initializer/random_uniform:08
�
block_13_expand_BN/gamma:0block_13_expand_BN/gamma/Assign.block_13_expand_BN/gamma/Read/ReadVariableOp:0(2+block_13_expand_BN/gamma/Initializer/ones:08
�
block_13_expand_BN/beta:0block_13_expand_BN/beta/Assign-block_13_expand_BN/beta/Read/ReadVariableOp:0(2+block_13_expand_BN/beta/Initializer/zeros:08
�
%block_13_depthwise/depthwise_kernel:0*block_13_depthwise/depthwise_kernel/Assign9block_13_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2@block_13_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_13_depthwise_BN/gamma:0"block_13_depthwise_BN/gamma/Assign1block_13_depthwise_BN/gamma/Read/ReadVariableOp:0(2.block_13_depthwise_BN/gamma/Initializer/ones:08
�
block_13_depthwise_BN/beta:0!block_13_depthwise_BN/beta/Assign0block_13_depthwise_BN/beta/Read/ReadVariableOp:0(2.block_13_depthwise_BN/beta/Initializer/zeros:08
�
block_13_project/kernel:0block_13_project/kernel/Assign-block_13_project/kernel/Read/ReadVariableOp:0(24block_13_project/kernel/Initializer/random_uniform:08
�
block_13_project_BN/gamma:0 block_13_project_BN/gamma/Assign/block_13_project_BN/gamma/Read/ReadVariableOp:0(2,block_13_project_BN/gamma/Initializer/ones:08
�
block_13_project_BN/beta:0block_13_project_BN/beta/Assign.block_13_project_BN/beta/Read/ReadVariableOp:0(2,block_13_project_BN/beta/Initializer/zeros:08
�
block_14_expand/kernel:0block_14_expand/kernel/Assign,block_14_expand/kernel/Read/ReadVariableOp:0(23block_14_expand/kernel/Initializer/random_uniform:08
�
block_14_expand_BN/gamma:0block_14_expand_BN/gamma/Assign.block_14_expand_BN/gamma/Read/ReadVariableOp:0(2+block_14_expand_BN/gamma/Initializer/ones:08
�
block_14_expand_BN/beta:0block_14_expand_BN/beta/Assign-block_14_expand_BN/beta/Read/ReadVariableOp:0(2+block_14_expand_BN/beta/Initializer/zeros:08
�
%block_14_depthwise/depthwise_kernel:0*block_14_depthwise/depthwise_kernel/Assign9block_14_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2@block_14_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_14_depthwise_BN/gamma:0"block_14_depthwise_BN/gamma/Assign1block_14_depthwise_BN/gamma/Read/ReadVariableOp:0(2.block_14_depthwise_BN/gamma/Initializer/ones:08
�
block_14_depthwise_BN/beta:0!block_14_depthwise_BN/beta/Assign0block_14_depthwise_BN/beta/Read/ReadVariableOp:0(2.block_14_depthwise_BN/beta/Initializer/zeros:08
�
block_14_project/kernel:0block_14_project/kernel/Assign-block_14_project/kernel/Read/ReadVariableOp:0(24block_14_project/kernel/Initializer/random_uniform:08
�
block_14_project_BN/gamma:0 block_14_project_BN/gamma/Assign/block_14_project_BN/gamma/Read/ReadVariableOp:0(2,block_14_project_BN/gamma/Initializer/ones:08
�
block_14_project_BN/beta:0block_14_project_BN/beta/Assign.block_14_project_BN/beta/Read/ReadVariableOp:0(2,block_14_project_BN/beta/Initializer/zeros:08
�
block_15_expand/kernel:0block_15_expand/kernel/Assign,block_15_expand/kernel/Read/ReadVariableOp:0(23block_15_expand/kernel/Initializer/random_uniform:08
�
block_15_expand_BN/gamma:0block_15_expand_BN/gamma/Assign.block_15_expand_BN/gamma/Read/ReadVariableOp:0(2+block_15_expand_BN/gamma/Initializer/ones:08
�
block_15_expand_BN/beta:0block_15_expand_BN/beta/Assign-block_15_expand_BN/beta/Read/ReadVariableOp:0(2+block_15_expand_BN/beta/Initializer/zeros:08
�
%block_15_depthwise/depthwise_kernel:0*block_15_depthwise/depthwise_kernel/Assign9block_15_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2@block_15_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_15_depthwise_BN/gamma:0"block_15_depthwise_BN/gamma/Assign1block_15_depthwise_BN/gamma/Read/ReadVariableOp:0(2.block_15_depthwise_BN/gamma/Initializer/ones:08
�
block_15_depthwise_BN/beta:0!block_15_depthwise_BN/beta/Assign0block_15_depthwise_BN/beta/Read/ReadVariableOp:0(2.block_15_depthwise_BN/beta/Initializer/zeros:08
�
block_15_project/kernel:0block_15_project/kernel/Assign-block_15_project/kernel/Read/ReadVariableOp:0(24block_15_project/kernel/Initializer/random_uniform:08
�
block_15_project_BN/gamma:0 block_15_project_BN/gamma/Assign/block_15_project_BN/gamma/Read/ReadVariableOp:0(2,block_15_project_BN/gamma/Initializer/ones:08
�
block_15_project_BN/beta:0block_15_project_BN/beta/Assign.block_15_project_BN/beta/Read/ReadVariableOp:0(2,block_15_project_BN/beta/Initializer/zeros:08
�
block_16_expand/kernel:0block_16_expand/kernel/Assign,block_16_expand/kernel/Read/ReadVariableOp:0(23block_16_expand/kernel/Initializer/random_uniform:08
�
block_16_expand_BN/gamma:0block_16_expand_BN/gamma/Assign.block_16_expand_BN/gamma/Read/ReadVariableOp:0(2+block_16_expand_BN/gamma/Initializer/ones:08
�
block_16_expand_BN/beta:0block_16_expand_BN/beta/Assign-block_16_expand_BN/beta/Read/ReadVariableOp:0(2+block_16_expand_BN/beta/Initializer/zeros:08
�
%block_16_depthwise/depthwise_kernel:0*block_16_depthwise/depthwise_kernel/Assign9block_16_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2@block_16_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_16_depthwise_BN/gamma:0"block_16_depthwise_BN/gamma/Assign1block_16_depthwise_BN/gamma/Read/ReadVariableOp:0(2.block_16_depthwise_BN/gamma/Initializer/ones:08
�
block_16_depthwise_BN/beta:0!block_16_depthwise_BN/beta/Assign0block_16_depthwise_BN/beta/Read/ReadVariableOp:0(2.block_16_depthwise_BN/beta/Initializer/zeros:08
�
block_16_project/kernel:0block_16_project/kernel/Assign-block_16_project/kernel/Read/ReadVariableOp:0(24block_16_project/kernel/Initializer/random_uniform:08
�
block_16_project_BN/gamma:0 block_16_project_BN/gamma/Assign/block_16_project_BN/gamma/Read/ReadVariableOp:0(2,block_16_project_BN/gamma/Initializer/ones:08
�
block_16_project_BN/beta:0block_16_project_BN/beta/Assign.block_16_project_BN/beta/Read/ReadVariableOp:0(2,block_16_project_BN/beta/Initializer/zeros:08
|
Conv_1/kernel:0Conv_1/kernel/Assign#Conv_1/kernel/Read/ReadVariableOp:0(2*Conv_1/kernel/Initializer/random_uniform:08
z
Conv_1_bn/gamma:0Conv_1_bn/gamma/Assign%Conv_1_bn/gamma/Read/ReadVariableOp:0(2"Conv_1_bn/gamma/Initializer/ones:08
w
Conv_1_bn/beta:0Conv_1_bn/beta/Assign$Conv_1_bn/beta/Read/ReadVariableOp:0(2"Conv_1_bn/beta/Initializer/zeros:08"��
	variables����
x
Conv1/kernel:0Conv1/kernel/Assign"Conv1/kernel/Read/ReadVariableOp:0(2)Conv1/kernel/Initializer/random_uniform:08
v
bn_Conv1/gamma:0bn_Conv1/gamma/Assign$bn_Conv1/gamma/Read/ReadVariableOp:0(2!bn_Conv1/gamma/Initializer/ones:08
s
bn_Conv1/beta:0bn_Conv1/beta/Assign#bn_Conv1/beta/Read/ReadVariableOp:0(2!bn_Conv1/beta/Initializer/zeros:08
�
bn_Conv1/moving_mean:0bn_Conv1/moving_mean/Assign*bn_Conv1/moving_mean/Read/ReadVariableOp:0(2(bn_Conv1/moving_mean/Initializer/zeros:0@H
�
bn_Conv1/moving_variance:0bn_Conv1/moving_variance/Assign.bn_Conv1/moving_variance/Read/ReadVariableOp:0(2+bn_Conv1/moving_variance/Initializer/ones:0@H
�
*expanded_conv_depthwise/depthwise_kernel:0/expanded_conv_depthwise/depthwise_kernel/Assign>expanded_conv_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2Eexpanded_conv_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
"expanded_conv_depthwise_BN/gamma:0'expanded_conv_depthwise_BN/gamma/Assign6expanded_conv_depthwise_BN/gamma/Read/ReadVariableOp:0(23expanded_conv_depthwise_BN/gamma/Initializer/ones:08
�
!expanded_conv_depthwise_BN/beta:0&expanded_conv_depthwise_BN/beta/Assign5expanded_conv_depthwise_BN/beta/Read/ReadVariableOp:0(23expanded_conv_depthwise_BN/beta/Initializer/zeros:08
�
(expanded_conv_depthwise_BN/moving_mean:0-expanded_conv_depthwise_BN/moving_mean/Assign<expanded_conv_depthwise_BN/moving_mean/Read/ReadVariableOp:0(2:expanded_conv_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
,expanded_conv_depthwise_BN/moving_variance:01expanded_conv_depthwise_BN/moving_variance/Assign@expanded_conv_depthwise_BN/moving_variance/Read/ReadVariableOp:0(2=expanded_conv_depthwise_BN/moving_variance/Initializer/ones:0@H
�
expanded_conv_project/kernel:0#expanded_conv_project/kernel/Assign2expanded_conv_project/kernel/Read/ReadVariableOp:0(29expanded_conv_project/kernel/Initializer/random_uniform:08
�
 expanded_conv_project_BN/gamma:0%expanded_conv_project_BN/gamma/Assign4expanded_conv_project_BN/gamma/Read/ReadVariableOp:0(21expanded_conv_project_BN/gamma/Initializer/ones:08
�
expanded_conv_project_BN/beta:0$expanded_conv_project_BN/beta/Assign3expanded_conv_project_BN/beta/Read/ReadVariableOp:0(21expanded_conv_project_BN/beta/Initializer/zeros:08
�
&expanded_conv_project_BN/moving_mean:0+expanded_conv_project_BN/moving_mean/Assign:expanded_conv_project_BN/moving_mean/Read/ReadVariableOp:0(28expanded_conv_project_BN/moving_mean/Initializer/zeros:0@H
�
*expanded_conv_project_BN/moving_variance:0/expanded_conv_project_BN/moving_variance/Assign>expanded_conv_project_BN/moving_variance/Read/ReadVariableOp:0(2;expanded_conv_project_BN/moving_variance/Initializer/ones:0@H
�
block_1_expand/kernel:0block_1_expand/kernel/Assign+block_1_expand/kernel/Read/ReadVariableOp:0(22block_1_expand/kernel/Initializer/random_uniform:08
�
block_1_expand_BN/gamma:0block_1_expand_BN/gamma/Assign-block_1_expand_BN/gamma/Read/ReadVariableOp:0(2*block_1_expand_BN/gamma/Initializer/ones:08
�
block_1_expand_BN/beta:0block_1_expand_BN/beta/Assign,block_1_expand_BN/beta/Read/ReadVariableOp:0(2*block_1_expand_BN/beta/Initializer/zeros:08
�
block_1_expand_BN/moving_mean:0$block_1_expand_BN/moving_mean/Assign3block_1_expand_BN/moving_mean/Read/ReadVariableOp:0(21block_1_expand_BN/moving_mean/Initializer/zeros:0@H
�
#block_1_expand_BN/moving_variance:0(block_1_expand_BN/moving_variance/Assign7block_1_expand_BN/moving_variance/Read/ReadVariableOp:0(24block_1_expand_BN/moving_variance/Initializer/ones:0@H
�
$block_1_depthwise/depthwise_kernel:0)block_1_depthwise/depthwise_kernel/Assign8block_1_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_1_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_1_depthwise_BN/gamma:0!block_1_depthwise_BN/gamma/Assign0block_1_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_1_depthwise_BN/gamma/Initializer/ones:08
�
block_1_depthwise_BN/beta:0 block_1_depthwise_BN/beta/Assign/block_1_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_1_depthwise_BN/beta/Initializer/zeros:08
�
"block_1_depthwise_BN/moving_mean:0'block_1_depthwise_BN/moving_mean/Assign6block_1_depthwise_BN/moving_mean/Read/ReadVariableOp:0(24block_1_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
&block_1_depthwise_BN/moving_variance:0+block_1_depthwise_BN/moving_variance/Assign:block_1_depthwise_BN/moving_variance/Read/ReadVariableOp:0(27block_1_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_1_project/kernel:0block_1_project/kernel/Assign,block_1_project/kernel/Read/ReadVariableOp:0(23block_1_project/kernel/Initializer/random_uniform:08
�
block_1_project_BN/gamma:0block_1_project_BN/gamma/Assign.block_1_project_BN/gamma/Read/ReadVariableOp:0(2+block_1_project_BN/gamma/Initializer/ones:08
�
block_1_project_BN/beta:0block_1_project_BN/beta/Assign-block_1_project_BN/beta/Read/ReadVariableOp:0(2+block_1_project_BN/beta/Initializer/zeros:08
�
 block_1_project_BN/moving_mean:0%block_1_project_BN/moving_mean/Assign4block_1_project_BN/moving_mean/Read/ReadVariableOp:0(22block_1_project_BN/moving_mean/Initializer/zeros:0@H
�
$block_1_project_BN/moving_variance:0)block_1_project_BN/moving_variance/Assign8block_1_project_BN/moving_variance/Read/ReadVariableOp:0(25block_1_project_BN/moving_variance/Initializer/ones:0@H
�
block_2_expand/kernel:0block_2_expand/kernel/Assign+block_2_expand/kernel/Read/ReadVariableOp:0(22block_2_expand/kernel/Initializer/random_uniform:08
�
block_2_expand_BN/gamma:0block_2_expand_BN/gamma/Assign-block_2_expand_BN/gamma/Read/ReadVariableOp:0(2*block_2_expand_BN/gamma/Initializer/ones:08
�
block_2_expand_BN/beta:0block_2_expand_BN/beta/Assign,block_2_expand_BN/beta/Read/ReadVariableOp:0(2*block_2_expand_BN/beta/Initializer/zeros:08
�
block_2_expand_BN/moving_mean:0$block_2_expand_BN/moving_mean/Assign3block_2_expand_BN/moving_mean/Read/ReadVariableOp:0(21block_2_expand_BN/moving_mean/Initializer/zeros:0@H
�
#block_2_expand_BN/moving_variance:0(block_2_expand_BN/moving_variance/Assign7block_2_expand_BN/moving_variance/Read/ReadVariableOp:0(24block_2_expand_BN/moving_variance/Initializer/ones:0@H
�
$block_2_depthwise/depthwise_kernel:0)block_2_depthwise/depthwise_kernel/Assign8block_2_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_2_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_2_depthwise_BN/gamma:0!block_2_depthwise_BN/gamma/Assign0block_2_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_2_depthwise_BN/gamma/Initializer/ones:08
�
block_2_depthwise_BN/beta:0 block_2_depthwise_BN/beta/Assign/block_2_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_2_depthwise_BN/beta/Initializer/zeros:08
�
"block_2_depthwise_BN/moving_mean:0'block_2_depthwise_BN/moving_mean/Assign6block_2_depthwise_BN/moving_mean/Read/ReadVariableOp:0(24block_2_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
&block_2_depthwise_BN/moving_variance:0+block_2_depthwise_BN/moving_variance/Assign:block_2_depthwise_BN/moving_variance/Read/ReadVariableOp:0(27block_2_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_2_project/kernel:0block_2_project/kernel/Assign,block_2_project/kernel/Read/ReadVariableOp:0(23block_2_project/kernel/Initializer/random_uniform:08
�
block_2_project_BN/gamma:0block_2_project_BN/gamma/Assign.block_2_project_BN/gamma/Read/ReadVariableOp:0(2+block_2_project_BN/gamma/Initializer/ones:08
�
block_2_project_BN/beta:0block_2_project_BN/beta/Assign-block_2_project_BN/beta/Read/ReadVariableOp:0(2+block_2_project_BN/beta/Initializer/zeros:08
�
 block_2_project_BN/moving_mean:0%block_2_project_BN/moving_mean/Assign4block_2_project_BN/moving_mean/Read/ReadVariableOp:0(22block_2_project_BN/moving_mean/Initializer/zeros:0@H
�
$block_2_project_BN/moving_variance:0)block_2_project_BN/moving_variance/Assign8block_2_project_BN/moving_variance/Read/ReadVariableOp:0(25block_2_project_BN/moving_variance/Initializer/ones:0@H
�
block_3_expand/kernel:0block_3_expand/kernel/Assign+block_3_expand/kernel/Read/ReadVariableOp:0(22block_3_expand/kernel/Initializer/random_uniform:08
�
block_3_expand_BN/gamma:0block_3_expand_BN/gamma/Assign-block_3_expand_BN/gamma/Read/ReadVariableOp:0(2*block_3_expand_BN/gamma/Initializer/ones:08
�
block_3_expand_BN/beta:0block_3_expand_BN/beta/Assign,block_3_expand_BN/beta/Read/ReadVariableOp:0(2*block_3_expand_BN/beta/Initializer/zeros:08
�
block_3_expand_BN/moving_mean:0$block_3_expand_BN/moving_mean/Assign3block_3_expand_BN/moving_mean/Read/ReadVariableOp:0(21block_3_expand_BN/moving_mean/Initializer/zeros:0@H
�
#block_3_expand_BN/moving_variance:0(block_3_expand_BN/moving_variance/Assign7block_3_expand_BN/moving_variance/Read/ReadVariableOp:0(24block_3_expand_BN/moving_variance/Initializer/ones:0@H
�
$block_3_depthwise/depthwise_kernel:0)block_3_depthwise/depthwise_kernel/Assign8block_3_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_3_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_3_depthwise_BN/gamma:0!block_3_depthwise_BN/gamma/Assign0block_3_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_3_depthwise_BN/gamma/Initializer/ones:08
�
block_3_depthwise_BN/beta:0 block_3_depthwise_BN/beta/Assign/block_3_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_3_depthwise_BN/beta/Initializer/zeros:08
�
"block_3_depthwise_BN/moving_mean:0'block_3_depthwise_BN/moving_mean/Assign6block_3_depthwise_BN/moving_mean/Read/ReadVariableOp:0(24block_3_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
&block_3_depthwise_BN/moving_variance:0+block_3_depthwise_BN/moving_variance/Assign:block_3_depthwise_BN/moving_variance/Read/ReadVariableOp:0(27block_3_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_3_project/kernel:0block_3_project/kernel/Assign,block_3_project/kernel/Read/ReadVariableOp:0(23block_3_project/kernel/Initializer/random_uniform:08
�
block_3_project_BN/gamma:0block_3_project_BN/gamma/Assign.block_3_project_BN/gamma/Read/ReadVariableOp:0(2+block_3_project_BN/gamma/Initializer/ones:08
�
block_3_project_BN/beta:0block_3_project_BN/beta/Assign-block_3_project_BN/beta/Read/ReadVariableOp:0(2+block_3_project_BN/beta/Initializer/zeros:08
�
 block_3_project_BN/moving_mean:0%block_3_project_BN/moving_mean/Assign4block_3_project_BN/moving_mean/Read/ReadVariableOp:0(22block_3_project_BN/moving_mean/Initializer/zeros:0@H
�
$block_3_project_BN/moving_variance:0)block_3_project_BN/moving_variance/Assign8block_3_project_BN/moving_variance/Read/ReadVariableOp:0(25block_3_project_BN/moving_variance/Initializer/ones:0@H
�
block_4_expand/kernel:0block_4_expand/kernel/Assign+block_4_expand/kernel/Read/ReadVariableOp:0(22block_4_expand/kernel/Initializer/random_uniform:08
�
block_4_expand_BN/gamma:0block_4_expand_BN/gamma/Assign-block_4_expand_BN/gamma/Read/ReadVariableOp:0(2*block_4_expand_BN/gamma/Initializer/ones:08
�
block_4_expand_BN/beta:0block_4_expand_BN/beta/Assign,block_4_expand_BN/beta/Read/ReadVariableOp:0(2*block_4_expand_BN/beta/Initializer/zeros:08
�
block_4_expand_BN/moving_mean:0$block_4_expand_BN/moving_mean/Assign3block_4_expand_BN/moving_mean/Read/ReadVariableOp:0(21block_4_expand_BN/moving_mean/Initializer/zeros:0@H
�
#block_4_expand_BN/moving_variance:0(block_4_expand_BN/moving_variance/Assign7block_4_expand_BN/moving_variance/Read/ReadVariableOp:0(24block_4_expand_BN/moving_variance/Initializer/ones:0@H
�
$block_4_depthwise/depthwise_kernel:0)block_4_depthwise/depthwise_kernel/Assign8block_4_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_4_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_4_depthwise_BN/gamma:0!block_4_depthwise_BN/gamma/Assign0block_4_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_4_depthwise_BN/gamma/Initializer/ones:08
�
block_4_depthwise_BN/beta:0 block_4_depthwise_BN/beta/Assign/block_4_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_4_depthwise_BN/beta/Initializer/zeros:08
�
"block_4_depthwise_BN/moving_mean:0'block_4_depthwise_BN/moving_mean/Assign6block_4_depthwise_BN/moving_mean/Read/ReadVariableOp:0(24block_4_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
&block_4_depthwise_BN/moving_variance:0+block_4_depthwise_BN/moving_variance/Assign:block_4_depthwise_BN/moving_variance/Read/ReadVariableOp:0(27block_4_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_4_project/kernel:0block_4_project/kernel/Assign,block_4_project/kernel/Read/ReadVariableOp:0(23block_4_project/kernel/Initializer/random_uniform:08
�
block_4_project_BN/gamma:0block_4_project_BN/gamma/Assign.block_4_project_BN/gamma/Read/ReadVariableOp:0(2+block_4_project_BN/gamma/Initializer/ones:08
�
block_4_project_BN/beta:0block_4_project_BN/beta/Assign-block_4_project_BN/beta/Read/ReadVariableOp:0(2+block_4_project_BN/beta/Initializer/zeros:08
�
 block_4_project_BN/moving_mean:0%block_4_project_BN/moving_mean/Assign4block_4_project_BN/moving_mean/Read/ReadVariableOp:0(22block_4_project_BN/moving_mean/Initializer/zeros:0@H
�
$block_4_project_BN/moving_variance:0)block_4_project_BN/moving_variance/Assign8block_4_project_BN/moving_variance/Read/ReadVariableOp:0(25block_4_project_BN/moving_variance/Initializer/ones:0@H
�
block_5_expand/kernel:0block_5_expand/kernel/Assign+block_5_expand/kernel/Read/ReadVariableOp:0(22block_5_expand/kernel/Initializer/random_uniform:08
�
block_5_expand_BN/gamma:0block_5_expand_BN/gamma/Assign-block_5_expand_BN/gamma/Read/ReadVariableOp:0(2*block_5_expand_BN/gamma/Initializer/ones:08
�
block_5_expand_BN/beta:0block_5_expand_BN/beta/Assign,block_5_expand_BN/beta/Read/ReadVariableOp:0(2*block_5_expand_BN/beta/Initializer/zeros:08
�
block_5_expand_BN/moving_mean:0$block_5_expand_BN/moving_mean/Assign3block_5_expand_BN/moving_mean/Read/ReadVariableOp:0(21block_5_expand_BN/moving_mean/Initializer/zeros:0@H
�
#block_5_expand_BN/moving_variance:0(block_5_expand_BN/moving_variance/Assign7block_5_expand_BN/moving_variance/Read/ReadVariableOp:0(24block_5_expand_BN/moving_variance/Initializer/ones:0@H
�
$block_5_depthwise/depthwise_kernel:0)block_5_depthwise/depthwise_kernel/Assign8block_5_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_5_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_5_depthwise_BN/gamma:0!block_5_depthwise_BN/gamma/Assign0block_5_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_5_depthwise_BN/gamma/Initializer/ones:08
�
block_5_depthwise_BN/beta:0 block_5_depthwise_BN/beta/Assign/block_5_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_5_depthwise_BN/beta/Initializer/zeros:08
�
"block_5_depthwise_BN/moving_mean:0'block_5_depthwise_BN/moving_mean/Assign6block_5_depthwise_BN/moving_mean/Read/ReadVariableOp:0(24block_5_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
&block_5_depthwise_BN/moving_variance:0+block_5_depthwise_BN/moving_variance/Assign:block_5_depthwise_BN/moving_variance/Read/ReadVariableOp:0(27block_5_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_5_project/kernel:0block_5_project/kernel/Assign,block_5_project/kernel/Read/ReadVariableOp:0(23block_5_project/kernel/Initializer/random_uniform:08
�
block_5_project_BN/gamma:0block_5_project_BN/gamma/Assign.block_5_project_BN/gamma/Read/ReadVariableOp:0(2+block_5_project_BN/gamma/Initializer/ones:08
�
block_5_project_BN/beta:0block_5_project_BN/beta/Assign-block_5_project_BN/beta/Read/ReadVariableOp:0(2+block_5_project_BN/beta/Initializer/zeros:08
�
 block_5_project_BN/moving_mean:0%block_5_project_BN/moving_mean/Assign4block_5_project_BN/moving_mean/Read/ReadVariableOp:0(22block_5_project_BN/moving_mean/Initializer/zeros:0@H
�
$block_5_project_BN/moving_variance:0)block_5_project_BN/moving_variance/Assign8block_5_project_BN/moving_variance/Read/ReadVariableOp:0(25block_5_project_BN/moving_variance/Initializer/ones:0@H
�
block_6_expand/kernel:0block_6_expand/kernel/Assign+block_6_expand/kernel/Read/ReadVariableOp:0(22block_6_expand/kernel/Initializer/random_uniform:08
�
block_6_expand_BN/gamma:0block_6_expand_BN/gamma/Assign-block_6_expand_BN/gamma/Read/ReadVariableOp:0(2*block_6_expand_BN/gamma/Initializer/ones:08
�
block_6_expand_BN/beta:0block_6_expand_BN/beta/Assign,block_6_expand_BN/beta/Read/ReadVariableOp:0(2*block_6_expand_BN/beta/Initializer/zeros:08
�
block_6_expand_BN/moving_mean:0$block_6_expand_BN/moving_mean/Assign3block_6_expand_BN/moving_mean/Read/ReadVariableOp:0(21block_6_expand_BN/moving_mean/Initializer/zeros:0@H
�
#block_6_expand_BN/moving_variance:0(block_6_expand_BN/moving_variance/Assign7block_6_expand_BN/moving_variance/Read/ReadVariableOp:0(24block_6_expand_BN/moving_variance/Initializer/ones:0@H
�
$block_6_depthwise/depthwise_kernel:0)block_6_depthwise/depthwise_kernel/Assign8block_6_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_6_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_6_depthwise_BN/gamma:0!block_6_depthwise_BN/gamma/Assign0block_6_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_6_depthwise_BN/gamma/Initializer/ones:08
�
block_6_depthwise_BN/beta:0 block_6_depthwise_BN/beta/Assign/block_6_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_6_depthwise_BN/beta/Initializer/zeros:08
�
"block_6_depthwise_BN/moving_mean:0'block_6_depthwise_BN/moving_mean/Assign6block_6_depthwise_BN/moving_mean/Read/ReadVariableOp:0(24block_6_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
&block_6_depthwise_BN/moving_variance:0+block_6_depthwise_BN/moving_variance/Assign:block_6_depthwise_BN/moving_variance/Read/ReadVariableOp:0(27block_6_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_6_project/kernel:0block_6_project/kernel/Assign,block_6_project/kernel/Read/ReadVariableOp:0(23block_6_project/kernel/Initializer/random_uniform:08
�
block_6_project_BN/gamma:0block_6_project_BN/gamma/Assign.block_6_project_BN/gamma/Read/ReadVariableOp:0(2+block_6_project_BN/gamma/Initializer/ones:08
�
block_6_project_BN/beta:0block_6_project_BN/beta/Assign-block_6_project_BN/beta/Read/ReadVariableOp:0(2+block_6_project_BN/beta/Initializer/zeros:08
�
 block_6_project_BN/moving_mean:0%block_6_project_BN/moving_mean/Assign4block_6_project_BN/moving_mean/Read/ReadVariableOp:0(22block_6_project_BN/moving_mean/Initializer/zeros:0@H
�
$block_6_project_BN/moving_variance:0)block_6_project_BN/moving_variance/Assign8block_6_project_BN/moving_variance/Read/ReadVariableOp:0(25block_6_project_BN/moving_variance/Initializer/ones:0@H
�
block_7_expand/kernel:0block_7_expand/kernel/Assign+block_7_expand/kernel/Read/ReadVariableOp:0(22block_7_expand/kernel/Initializer/random_uniform:08
�
block_7_expand_BN/gamma:0block_7_expand_BN/gamma/Assign-block_7_expand_BN/gamma/Read/ReadVariableOp:0(2*block_7_expand_BN/gamma/Initializer/ones:08
�
block_7_expand_BN/beta:0block_7_expand_BN/beta/Assign,block_7_expand_BN/beta/Read/ReadVariableOp:0(2*block_7_expand_BN/beta/Initializer/zeros:08
�
block_7_expand_BN/moving_mean:0$block_7_expand_BN/moving_mean/Assign3block_7_expand_BN/moving_mean/Read/ReadVariableOp:0(21block_7_expand_BN/moving_mean/Initializer/zeros:0@H
�
#block_7_expand_BN/moving_variance:0(block_7_expand_BN/moving_variance/Assign7block_7_expand_BN/moving_variance/Read/ReadVariableOp:0(24block_7_expand_BN/moving_variance/Initializer/ones:0@H
�
$block_7_depthwise/depthwise_kernel:0)block_7_depthwise/depthwise_kernel/Assign8block_7_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_7_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_7_depthwise_BN/gamma:0!block_7_depthwise_BN/gamma/Assign0block_7_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_7_depthwise_BN/gamma/Initializer/ones:08
�
block_7_depthwise_BN/beta:0 block_7_depthwise_BN/beta/Assign/block_7_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_7_depthwise_BN/beta/Initializer/zeros:08
�
"block_7_depthwise_BN/moving_mean:0'block_7_depthwise_BN/moving_mean/Assign6block_7_depthwise_BN/moving_mean/Read/ReadVariableOp:0(24block_7_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
&block_7_depthwise_BN/moving_variance:0+block_7_depthwise_BN/moving_variance/Assign:block_7_depthwise_BN/moving_variance/Read/ReadVariableOp:0(27block_7_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_7_project/kernel:0block_7_project/kernel/Assign,block_7_project/kernel/Read/ReadVariableOp:0(23block_7_project/kernel/Initializer/random_uniform:08
�
block_7_project_BN/gamma:0block_7_project_BN/gamma/Assign.block_7_project_BN/gamma/Read/ReadVariableOp:0(2+block_7_project_BN/gamma/Initializer/ones:08
�
block_7_project_BN/beta:0block_7_project_BN/beta/Assign-block_7_project_BN/beta/Read/ReadVariableOp:0(2+block_7_project_BN/beta/Initializer/zeros:08
�
 block_7_project_BN/moving_mean:0%block_7_project_BN/moving_mean/Assign4block_7_project_BN/moving_mean/Read/ReadVariableOp:0(22block_7_project_BN/moving_mean/Initializer/zeros:0@H
�
$block_7_project_BN/moving_variance:0)block_7_project_BN/moving_variance/Assign8block_7_project_BN/moving_variance/Read/ReadVariableOp:0(25block_7_project_BN/moving_variance/Initializer/ones:0@H
�
block_8_expand/kernel:0block_8_expand/kernel/Assign+block_8_expand/kernel/Read/ReadVariableOp:0(22block_8_expand/kernel/Initializer/random_uniform:08
�
block_8_expand_BN/gamma:0block_8_expand_BN/gamma/Assign-block_8_expand_BN/gamma/Read/ReadVariableOp:0(2*block_8_expand_BN/gamma/Initializer/ones:08
�
block_8_expand_BN/beta:0block_8_expand_BN/beta/Assign,block_8_expand_BN/beta/Read/ReadVariableOp:0(2*block_8_expand_BN/beta/Initializer/zeros:08
�
block_8_expand_BN/moving_mean:0$block_8_expand_BN/moving_mean/Assign3block_8_expand_BN/moving_mean/Read/ReadVariableOp:0(21block_8_expand_BN/moving_mean/Initializer/zeros:0@H
�
#block_8_expand_BN/moving_variance:0(block_8_expand_BN/moving_variance/Assign7block_8_expand_BN/moving_variance/Read/ReadVariableOp:0(24block_8_expand_BN/moving_variance/Initializer/ones:0@H
�
$block_8_depthwise/depthwise_kernel:0)block_8_depthwise/depthwise_kernel/Assign8block_8_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_8_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_8_depthwise_BN/gamma:0!block_8_depthwise_BN/gamma/Assign0block_8_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_8_depthwise_BN/gamma/Initializer/ones:08
�
block_8_depthwise_BN/beta:0 block_8_depthwise_BN/beta/Assign/block_8_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_8_depthwise_BN/beta/Initializer/zeros:08
�
"block_8_depthwise_BN/moving_mean:0'block_8_depthwise_BN/moving_mean/Assign6block_8_depthwise_BN/moving_mean/Read/ReadVariableOp:0(24block_8_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
&block_8_depthwise_BN/moving_variance:0+block_8_depthwise_BN/moving_variance/Assign:block_8_depthwise_BN/moving_variance/Read/ReadVariableOp:0(27block_8_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_8_project/kernel:0block_8_project/kernel/Assign,block_8_project/kernel/Read/ReadVariableOp:0(23block_8_project/kernel/Initializer/random_uniform:08
�
block_8_project_BN/gamma:0block_8_project_BN/gamma/Assign.block_8_project_BN/gamma/Read/ReadVariableOp:0(2+block_8_project_BN/gamma/Initializer/ones:08
�
block_8_project_BN/beta:0block_8_project_BN/beta/Assign-block_8_project_BN/beta/Read/ReadVariableOp:0(2+block_8_project_BN/beta/Initializer/zeros:08
�
 block_8_project_BN/moving_mean:0%block_8_project_BN/moving_mean/Assign4block_8_project_BN/moving_mean/Read/ReadVariableOp:0(22block_8_project_BN/moving_mean/Initializer/zeros:0@H
�
$block_8_project_BN/moving_variance:0)block_8_project_BN/moving_variance/Assign8block_8_project_BN/moving_variance/Read/ReadVariableOp:0(25block_8_project_BN/moving_variance/Initializer/ones:0@H
�
block_9_expand/kernel:0block_9_expand/kernel/Assign+block_9_expand/kernel/Read/ReadVariableOp:0(22block_9_expand/kernel/Initializer/random_uniform:08
�
block_9_expand_BN/gamma:0block_9_expand_BN/gamma/Assign-block_9_expand_BN/gamma/Read/ReadVariableOp:0(2*block_9_expand_BN/gamma/Initializer/ones:08
�
block_9_expand_BN/beta:0block_9_expand_BN/beta/Assign,block_9_expand_BN/beta/Read/ReadVariableOp:0(2*block_9_expand_BN/beta/Initializer/zeros:08
�
block_9_expand_BN/moving_mean:0$block_9_expand_BN/moving_mean/Assign3block_9_expand_BN/moving_mean/Read/ReadVariableOp:0(21block_9_expand_BN/moving_mean/Initializer/zeros:0@H
�
#block_9_expand_BN/moving_variance:0(block_9_expand_BN/moving_variance/Assign7block_9_expand_BN/moving_variance/Read/ReadVariableOp:0(24block_9_expand_BN/moving_variance/Initializer/ones:0@H
�
$block_9_depthwise/depthwise_kernel:0)block_9_depthwise/depthwise_kernel/Assign8block_9_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2?block_9_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_9_depthwise_BN/gamma:0!block_9_depthwise_BN/gamma/Assign0block_9_depthwise_BN/gamma/Read/ReadVariableOp:0(2-block_9_depthwise_BN/gamma/Initializer/ones:08
�
block_9_depthwise_BN/beta:0 block_9_depthwise_BN/beta/Assign/block_9_depthwise_BN/beta/Read/ReadVariableOp:0(2-block_9_depthwise_BN/beta/Initializer/zeros:08
�
"block_9_depthwise_BN/moving_mean:0'block_9_depthwise_BN/moving_mean/Assign6block_9_depthwise_BN/moving_mean/Read/ReadVariableOp:0(24block_9_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
&block_9_depthwise_BN/moving_variance:0+block_9_depthwise_BN/moving_variance/Assign:block_9_depthwise_BN/moving_variance/Read/ReadVariableOp:0(27block_9_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_9_project/kernel:0block_9_project/kernel/Assign,block_9_project/kernel/Read/ReadVariableOp:0(23block_9_project/kernel/Initializer/random_uniform:08
�
block_9_project_BN/gamma:0block_9_project_BN/gamma/Assign.block_9_project_BN/gamma/Read/ReadVariableOp:0(2+block_9_project_BN/gamma/Initializer/ones:08
�
block_9_project_BN/beta:0block_9_project_BN/beta/Assign-block_9_project_BN/beta/Read/ReadVariableOp:0(2+block_9_project_BN/beta/Initializer/zeros:08
�
 block_9_project_BN/moving_mean:0%block_9_project_BN/moving_mean/Assign4block_9_project_BN/moving_mean/Read/ReadVariableOp:0(22block_9_project_BN/moving_mean/Initializer/zeros:0@H
�
$block_9_project_BN/moving_variance:0)block_9_project_BN/moving_variance/Assign8block_9_project_BN/moving_variance/Read/ReadVariableOp:0(25block_9_project_BN/moving_variance/Initializer/ones:0@H
�
block_10_expand/kernel:0block_10_expand/kernel/Assign,block_10_expand/kernel/Read/ReadVariableOp:0(23block_10_expand/kernel/Initializer/random_uniform:08
�
block_10_expand_BN/gamma:0block_10_expand_BN/gamma/Assign.block_10_expand_BN/gamma/Read/ReadVariableOp:0(2+block_10_expand_BN/gamma/Initializer/ones:08
�
block_10_expand_BN/beta:0block_10_expand_BN/beta/Assign-block_10_expand_BN/beta/Read/ReadVariableOp:0(2+block_10_expand_BN/beta/Initializer/zeros:08
�
 block_10_expand_BN/moving_mean:0%block_10_expand_BN/moving_mean/Assign4block_10_expand_BN/moving_mean/Read/ReadVariableOp:0(22block_10_expand_BN/moving_mean/Initializer/zeros:0@H
�
$block_10_expand_BN/moving_variance:0)block_10_expand_BN/moving_variance/Assign8block_10_expand_BN/moving_variance/Read/ReadVariableOp:0(25block_10_expand_BN/moving_variance/Initializer/ones:0@H
�
%block_10_depthwise/depthwise_kernel:0*block_10_depthwise/depthwise_kernel/Assign9block_10_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2@block_10_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_10_depthwise_BN/gamma:0"block_10_depthwise_BN/gamma/Assign1block_10_depthwise_BN/gamma/Read/ReadVariableOp:0(2.block_10_depthwise_BN/gamma/Initializer/ones:08
�
block_10_depthwise_BN/beta:0!block_10_depthwise_BN/beta/Assign0block_10_depthwise_BN/beta/Read/ReadVariableOp:0(2.block_10_depthwise_BN/beta/Initializer/zeros:08
�
#block_10_depthwise_BN/moving_mean:0(block_10_depthwise_BN/moving_mean/Assign7block_10_depthwise_BN/moving_mean/Read/ReadVariableOp:0(25block_10_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
'block_10_depthwise_BN/moving_variance:0,block_10_depthwise_BN/moving_variance/Assign;block_10_depthwise_BN/moving_variance/Read/ReadVariableOp:0(28block_10_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_10_project/kernel:0block_10_project/kernel/Assign-block_10_project/kernel/Read/ReadVariableOp:0(24block_10_project/kernel/Initializer/random_uniform:08
�
block_10_project_BN/gamma:0 block_10_project_BN/gamma/Assign/block_10_project_BN/gamma/Read/ReadVariableOp:0(2,block_10_project_BN/gamma/Initializer/ones:08
�
block_10_project_BN/beta:0block_10_project_BN/beta/Assign.block_10_project_BN/beta/Read/ReadVariableOp:0(2,block_10_project_BN/beta/Initializer/zeros:08
�
!block_10_project_BN/moving_mean:0&block_10_project_BN/moving_mean/Assign5block_10_project_BN/moving_mean/Read/ReadVariableOp:0(23block_10_project_BN/moving_mean/Initializer/zeros:0@H
�
%block_10_project_BN/moving_variance:0*block_10_project_BN/moving_variance/Assign9block_10_project_BN/moving_variance/Read/ReadVariableOp:0(26block_10_project_BN/moving_variance/Initializer/ones:0@H
�
block_11_expand/kernel:0block_11_expand/kernel/Assign,block_11_expand/kernel/Read/ReadVariableOp:0(23block_11_expand/kernel/Initializer/random_uniform:08
�
block_11_expand_BN/gamma:0block_11_expand_BN/gamma/Assign.block_11_expand_BN/gamma/Read/ReadVariableOp:0(2+block_11_expand_BN/gamma/Initializer/ones:08
�
block_11_expand_BN/beta:0block_11_expand_BN/beta/Assign-block_11_expand_BN/beta/Read/ReadVariableOp:0(2+block_11_expand_BN/beta/Initializer/zeros:08
�
 block_11_expand_BN/moving_mean:0%block_11_expand_BN/moving_mean/Assign4block_11_expand_BN/moving_mean/Read/ReadVariableOp:0(22block_11_expand_BN/moving_mean/Initializer/zeros:0@H
�
$block_11_expand_BN/moving_variance:0)block_11_expand_BN/moving_variance/Assign8block_11_expand_BN/moving_variance/Read/ReadVariableOp:0(25block_11_expand_BN/moving_variance/Initializer/ones:0@H
�
%block_11_depthwise/depthwise_kernel:0*block_11_depthwise/depthwise_kernel/Assign9block_11_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2@block_11_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_11_depthwise_BN/gamma:0"block_11_depthwise_BN/gamma/Assign1block_11_depthwise_BN/gamma/Read/ReadVariableOp:0(2.block_11_depthwise_BN/gamma/Initializer/ones:08
�
block_11_depthwise_BN/beta:0!block_11_depthwise_BN/beta/Assign0block_11_depthwise_BN/beta/Read/ReadVariableOp:0(2.block_11_depthwise_BN/beta/Initializer/zeros:08
�
#block_11_depthwise_BN/moving_mean:0(block_11_depthwise_BN/moving_mean/Assign7block_11_depthwise_BN/moving_mean/Read/ReadVariableOp:0(25block_11_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
'block_11_depthwise_BN/moving_variance:0,block_11_depthwise_BN/moving_variance/Assign;block_11_depthwise_BN/moving_variance/Read/ReadVariableOp:0(28block_11_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_11_project/kernel:0block_11_project/kernel/Assign-block_11_project/kernel/Read/ReadVariableOp:0(24block_11_project/kernel/Initializer/random_uniform:08
�
block_11_project_BN/gamma:0 block_11_project_BN/gamma/Assign/block_11_project_BN/gamma/Read/ReadVariableOp:0(2,block_11_project_BN/gamma/Initializer/ones:08
�
block_11_project_BN/beta:0block_11_project_BN/beta/Assign.block_11_project_BN/beta/Read/ReadVariableOp:0(2,block_11_project_BN/beta/Initializer/zeros:08
�
!block_11_project_BN/moving_mean:0&block_11_project_BN/moving_mean/Assign5block_11_project_BN/moving_mean/Read/ReadVariableOp:0(23block_11_project_BN/moving_mean/Initializer/zeros:0@H
�
%block_11_project_BN/moving_variance:0*block_11_project_BN/moving_variance/Assign9block_11_project_BN/moving_variance/Read/ReadVariableOp:0(26block_11_project_BN/moving_variance/Initializer/ones:0@H
�
block_12_expand/kernel:0block_12_expand/kernel/Assign,block_12_expand/kernel/Read/ReadVariableOp:0(23block_12_expand/kernel/Initializer/random_uniform:08
�
block_12_expand_BN/gamma:0block_12_expand_BN/gamma/Assign.block_12_expand_BN/gamma/Read/ReadVariableOp:0(2+block_12_expand_BN/gamma/Initializer/ones:08
�
block_12_expand_BN/beta:0block_12_expand_BN/beta/Assign-block_12_expand_BN/beta/Read/ReadVariableOp:0(2+block_12_expand_BN/beta/Initializer/zeros:08
�
 block_12_expand_BN/moving_mean:0%block_12_expand_BN/moving_mean/Assign4block_12_expand_BN/moving_mean/Read/ReadVariableOp:0(22block_12_expand_BN/moving_mean/Initializer/zeros:0@H
�
$block_12_expand_BN/moving_variance:0)block_12_expand_BN/moving_variance/Assign8block_12_expand_BN/moving_variance/Read/ReadVariableOp:0(25block_12_expand_BN/moving_variance/Initializer/ones:0@H
�
%block_12_depthwise/depthwise_kernel:0*block_12_depthwise/depthwise_kernel/Assign9block_12_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2@block_12_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_12_depthwise_BN/gamma:0"block_12_depthwise_BN/gamma/Assign1block_12_depthwise_BN/gamma/Read/ReadVariableOp:0(2.block_12_depthwise_BN/gamma/Initializer/ones:08
�
block_12_depthwise_BN/beta:0!block_12_depthwise_BN/beta/Assign0block_12_depthwise_BN/beta/Read/ReadVariableOp:0(2.block_12_depthwise_BN/beta/Initializer/zeros:08
�
#block_12_depthwise_BN/moving_mean:0(block_12_depthwise_BN/moving_mean/Assign7block_12_depthwise_BN/moving_mean/Read/ReadVariableOp:0(25block_12_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
'block_12_depthwise_BN/moving_variance:0,block_12_depthwise_BN/moving_variance/Assign;block_12_depthwise_BN/moving_variance/Read/ReadVariableOp:0(28block_12_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_12_project/kernel:0block_12_project/kernel/Assign-block_12_project/kernel/Read/ReadVariableOp:0(24block_12_project/kernel/Initializer/random_uniform:08
�
block_12_project_BN/gamma:0 block_12_project_BN/gamma/Assign/block_12_project_BN/gamma/Read/ReadVariableOp:0(2,block_12_project_BN/gamma/Initializer/ones:08
�
block_12_project_BN/beta:0block_12_project_BN/beta/Assign.block_12_project_BN/beta/Read/ReadVariableOp:0(2,block_12_project_BN/beta/Initializer/zeros:08
�
!block_12_project_BN/moving_mean:0&block_12_project_BN/moving_mean/Assign5block_12_project_BN/moving_mean/Read/ReadVariableOp:0(23block_12_project_BN/moving_mean/Initializer/zeros:0@H
�
%block_12_project_BN/moving_variance:0*block_12_project_BN/moving_variance/Assign9block_12_project_BN/moving_variance/Read/ReadVariableOp:0(26block_12_project_BN/moving_variance/Initializer/ones:0@H
�
block_13_expand/kernel:0block_13_expand/kernel/Assign,block_13_expand/kernel/Read/ReadVariableOp:0(23block_13_expand/kernel/Initializer/random_uniform:08
�
block_13_expand_BN/gamma:0block_13_expand_BN/gamma/Assign.block_13_expand_BN/gamma/Read/ReadVariableOp:0(2+block_13_expand_BN/gamma/Initializer/ones:08
�
block_13_expand_BN/beta:0block_13_expand_BN/beta/Assign-block_13_expand_BN/beta/Read/ReadVariableOp:0(2+block_13_expand_BN/beta/Initializer/zeros:08
�
 block_13_expand_BN/moving_mean:0%block_13_expand_BN/moving_mean/Assign4block_13_expand_BN/moving_mean/Read/ReadVariableOp:0(22block_13_expand_BN/moving_mean/Initializer/zeros:0@H
�
$block_13_expand_BN/moving_variance:0)block_13_expand_BN/moving_variance/Assign8block_13_expand_BN/moving_variance/Read/ReadVariableOp:0(25block_13_expand_BN/moving_variance/Initializer/ones:0@H
�
%block_13_depthwise/depthwise_kernel:0*block_13_depthwise/depthwise_kernel/Assign9block_13_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2@block_13_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_13_depthwise_BN/gamma:0"block_13_depthwise_BN/gamma/Assign1block_13_depthwise_BN/gamma/Read/ReadVariableOp:0(2.block_13_depthwise_BN/gamma/Initializer/ones:08
�
block_13_depthwise_BN/beta:0!block_13_depthwise_BN/beta/Assign0block_13_depthwise_BN/beta/Read/ReadVariableOp:0(2.block_13_depthwise_BN/beta/Initializer/zeros:08
�
#block_13_depthwise_BN/moving_mean:0(block_13_depthwise_BN/moving_mean/Assign7block_13_depthwise_BN/moving_mean/Read/ReadVariableOp:0(25block_13_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
'block_13_depthwise_BN/moving_variance:0,block_13_depthwise_BN/moving_variance/Assign;block_13_depthwise_BN/moving_variance/Read/ReadVariableOp:0(28block_13_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_13_project/kernel:0block_13_project/kernel/Assign-block_13_project/kernel/Read/ReadVariableOp:0(24block_13_project/kernel/Initializer/random_uniform:08
�
block_13_project_BN/gamma:0 block_13_project_BN/gamma/Assign/block_13_project_BN/gamma/Read/ReadVariableOp:0(2,block_13_project_BN/gamma/Initializer/ones:08
�
block_13_project_BN/beta:0block_13_project_BN/beta/Assign.block_13_project_BN/beta/Read/ReadVariableOp:0(2,block_13_project_BN/beta/Initializer/zeros:08
�
!block_13_project_BN/moving_mean:0&block_13_project_BN/moving_mean/Assign5block_13_project_BN/moving_mean/Read/ReadVariableOp:0(23block_13_project_BN/moving_mean/Initializer/zeros:0@H
�
%block_13_project_BN/moving_variance:0*block_13_project_BN/moving_variance/Assign9block_13_project_BN/moving_variance/Read/ReadVariableOp:0(26block_13_project_BN/moving_variance/Initializer/ones:0@H
�
block_14_expand/kernel:0block_14_expand/kernel/Assign,block_14_expand/kernel/Read/ReadVariableOp:0(23block_14_expand/kernel/Initializer/random_uniform:08
�
block_14_expand_BN/gamma:0block_14_expand_BN/gamma/Assign.block_14_expand_BN/gamma/Read/ReadVariableOp:0(2+block_14_expand_BN/gamma/Initializer/ones:08
�
block_14_expand_BN/beta:0block_14_expand_BN/beta/Assign-block_14_expand_BN/beta/Read/ReadVariableOp:0(2+block_14_expand_BN/beta/Initializer/zeros:08
�
 block_14_expand_BN/moving_mean:0%block_14_expand_BN/moving_mean/Assign4block_14_expand_BN/moving_mean/Read/ReadVariableOp:0(22block_14_expand_BN/moving_mean/Initializer/zeros:0@H
�
$block_14_expand_BN/moving_variance:0)block_14_expand_BN/moving_variance/Assign8block_14_expand_BN/moving_variance/Read/ReadVariableOp:0(25block_14_expand_BN/moving_variance/Initializer/ones:0@H
�
%block_14_depthwise/depthwise_kernel:0*block_14_depthwise/depthwise_kernel/Assign9block_14_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2@block_14_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_14_depthwise_BN/gamma:0"block_14_depthwise_BN/gamma/Assign1block_14_depthwise_BN/gamma/Read/ReadVariableOp:0(2.block_14_depthwise_BN/gamma/Initializer/ones:08
�
block_14_depthwise_BN/beta:0!block_14_depthwise_BN/beta/Assign0block_14_depthwise_BN/beta/Read/ReadVariableOp:0(2.block_14_depthwise_BN/beta/Initializer/zeros:08
�
#block_14_depthwise_BN/moving_mean:0(block_14_depthwise_BN/moving_mean/Assign7block_14_depthwise_BN/moving_mean/Read/ReadVariableOp:0(25block_14_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
'block_14_depthwise_BN/moving_variance:0,block_14_depthwise_BN/moving_variance/Assign;block_14_depthwise_BN/moving_variance/Read/ReadVariableOp:0(28block_14_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_14_project/kernel:0block_14_project/kernel/Assign-block_14_project/kernel/Read/ReadVariableOp:0(24block_14_project/kernel/Initializer/random_uniform:08
�
block_14_project_BN/gamma:0 block_14_project_BN/gamma/Assign/block_14_project_BN/gamma/Read/ReadVariableOp:0(2,block_14_project_BN/gamma/Initializer/ones:08
�
block_14_project_BN/beta:0block_14_project_BN/beta/Assign.block_14_project_BN/beta/Read/ReadVariableOp:0(2,block_14_project_BN/beta/Initializer/zeros:08
�
!block_14_project_BN/moving_mean:0&block_14_project_BN/moving_mean/Assign5block_14_project_BN/moving_mean/Read/ReadVariableOp:0(23block_14_project_BN/moving_mean/Initializer/zeros:0@H
�
%block_14_project_BN/moving_variance:0*block_14_project_BN/moving_variance/Assign9block_14_project_BN/moving_variance/Read/ReadVariableOp:0(26block_14_project_BN/moving_variance/Initializer/ones:0@H
�
block_15_expand/kernel:0block_15_expand/kernel/Assign,block_15_expand/kernel/Read/ReadVariableOp:0(23block_15_expand/kernel/Initializer/random_uniform:08
�
block_15_expand_BN/gamma:0block_15_expand_BN/gamma/Assign.block_15_expand_BN/gamma/Read/ReadVariableOp:0(2+block_15_expand_BN/gamma/Initializer/ones:08
�
block_15_expand_BN/beta:0block_15_expand_BN/beta/Assign-block_15_expand_BN/beta/Read/ReadVariableOp:0(2+block_15_expand_BN/beta/Initializer/zeros:08
�
 block_15_expand_BN/moving_mean:0%block_15_expand_BN/moving_mean/Assign4block_15_expand_BN/moving_mean/Read/ReadVariableOp:0(22block_15_expand_BN/moving_mean/Initializer/zeros:0@H
�
$block_15_expand_BN/moving_variance:0)block_15_expand_BN/moving_variance/Assign8block_15_expand_BN/moving_variance/Read/ReadVariableOp:0(25block_15_expand_BN/moving_variance/Initializer/ones:0@H
�
%block_15_depthwise/depthwise_kernel:0*block_15_depthwise/depthwise_kernel/Assign9block_15_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2@block_15_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_15_depthwise_BN/gamma:0"block_15_depthwise_BN/gamma/Assign1block_15_depthwise_BN/gamma/Read/ReadVariableOp:0(2.block_15_depthwise_BN/gamma/Initializer/ones:08
�
block_15_depthwise_BN/beta:0!block_15_depthwise_BN/beta/Assign0block_15_depthwise_BN/beta/Read/ReadVariableOp:0(2.block_15_depthwise_BN/beta/Initializer/zeros:08
�
#block_15_depthwise_BN/moving_mean:0(block_15_depthwise_BN/moving_mean/Assign7block_15_depthwise_BN/moving_mean/Read/ReadVariableOp:0(25block_15_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
'block_15_depthwise_BN/moving_variance:0,block_15_depthwise_BN/moving_variance/Assign;block_15_depthwise_BN/moving_variance/Read/ReadVariableOp:0(28block_15_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_15_project/kernel:0block_15_project/kernel/Assign-block_15_project/kernel/Read/ReadVariableOp:0(24block_15_project/kernel/Initializer/random_uniform:08
�
block_15_project_BN/gamma:0 block_15_project_BN/gamma/Assign/block_15_project_BN/gamma/Read/ReadVariableOp:0(2,block_15_project_BN/gamma/Initializer/ones:08
�
block_15_project_BN/beta:0block_15_project_BN/beta/Assign.block_15_project_BN/beta/Read/ReadVariableOp:0(2,block_15_project_BN/beta/Initializer/zeros:08
�
!block_15_project_BN/moving_mean:0&block_15_project_BN/moving_mean/Assign5block_15_project_BN/moving_mean/Read/ReadVariableOp:0(23block_15_project_BN/moving_mean/Initializer/zeros:0@H
�
%block_15_project_BN/moving_variance:0*block_15_project_BN/moving_variance/Assign9block_15_project_BN/moving_variance/Read/ReadVariableOp:0(26block_15_project_BN/moving_variance/Initializer/ones:0@H
�
block_16_expand/kernel:0block_16_expand/kernel/Assign,block_16_expand/kernel/Read/ReadVariableOp:0(23block_16_expand/kernel/Initializer/random_uniform:08
�
block_16_expand_BN/gamma:0block_16_expand_BN/gamma/Assign.block_16_expand_BN/gamma/Read/ReadVariableOp:0(2+block_16_expand_BN/gamma/Initializer/ones:08
�
block_16_expand_BN/beta:0block_16_expand_BN/beta/Assign-block_16_expand_BN/beta/Read/ReadVariableOp:0(2+block_16_expand_BN/beta/Initializer/zeros:08
�
 block_16_expand_BN/moving_mean:0%block_16_expand_BN/moving_mean/Assign4block_16_expand_BN/moving_mean/Read/ReadVariableOp:0(22block_16_expand_BN/moving_mean/Initializer/zeros:0@H
�
$block_16_expand_BN/moving_variance:0)block_16_expand_BN/moving_variance/Assign8block_16_expand_BN/moving_variance/Read/ReadVariableOp:0(25block_16_expand_BN/moving_variance/Initializer/ones:0@H
�
%block_16_depthwise/depthwise_kernel:0*block_16_depthwise/depthwise_kernel/Assign9block_16_depthwise/depthwise_kernel/Read/ReadVariableOp:0(2@block_16_depthwise/depthwise_kernel/Initializer/random_uniform:08
�
block_16_depthwise_BN/gamma:0"block_16_depthwise_BN/gamma/Assign1block_16_depthwise_BN/gamma/Read/ReadVariableOp:0(2.block_16_depthwise_BN/gamma/Initializer/ones:08
�
block_16_depthwise_BN/beta:0!block_16_depthwise_BN/beta/Assign0block_16_depthwise_BN/beta/Read/ReadVariableOp:0(2.block_16_depthwise_BN/beta/Initializer/zeros:08
�
#block_16_depthwise_BN/moving_mean:0(block_16_depthwise_BN/moving_mean/Assign7block_16_depthwise_BN/moving_mean/Read/ReadVariableOp:0(25block_16_depthwise_BN/moving_mean/Initializer/zeros:0@H
�
'block_16_depthwise_BN/moving_variance:0,block_16_depthwise_BN/moving_variance/Assign;block_16_depthwise_BN/moving_variance/Read/ReadVariableOp:0(28block_16_depthwise_BN/moving_variance/Initializer/ones:0@H
�
block_16_project/kernel:0block_16_project/kernel/Assign-block_16_project/kernel/Read/ReadVariableOp:0(24block_16_project/kernel/Initializer/random_uniform:08
�
block_16_project_BN/gamma:0 block_16_project_BN/gamma/Assign/block_16_project_BN/gamma/Read/ReadVariableOp:0(2,block_16_project_BN/gamma/Initializer/ones:08
�
block_16_project_BN/beta:0block_16_project_BN/beta/Assign.block_16_project_BN/beta/Read/ReadVariableOp:0(2,block_16_project_BN/beta/Initializer/zeros:08
�
!block_16_project_BN/moving_mean:0&block_16_project_BN/moving_mean/Assign5block_16_project_BN/moving_mean/Read/ReadVariableOp:0(23block_16_project_BN/moving_mean/Initializer/zeros:0@H
�
%block_16_project_BN/moving_variance:0*block_16_project_BN/moving_variance/Assign9block_16_project_BN/moving_variance/Read/ReadVariableOp:0(26block_16_project_BN/moving_variance/Initializer/ones:0@H
|
Conv_1/kernel:0Conv_1/kernel/Assign#Conv_1/kernel/Read/ReadVariableOp:0(2*Conv_1/kernel/Initializer/random_uniform:08
z
Conv_1_bn/gamma:0Conv_1_bn/gamma/Assign%Conv_1_bn/gamma/Read/ReadVariableOp:0(2"Conv_1_bn/gamma/Initializer/ones:08
w
Conv_1_bn/beta:0Conv_1_bn/beta/Assign$Conv_1_bn/beta/Read/ReadVariableOp:0(2"Conv_1_bn/beta/Initializer/zeros:08
�
Conv_1_bn/moving_mean:0Conv_1_bn/moving_mean/Assign+Conv_1_bn/moving_mean/Read/ReadVariableOp:0(2)Conv_1_bn/moving_mean/Initializer/zeros:0@H
�
Conv_1_bn/moving_variance:0 Conv_1_bn/moving_variance/Assign/Conv_1_bn/moving_variance/Read/ReadVariableOp:0(2,Conv_1_bn/moving_variance/Initializer/ones:0@H*�
serving_default�
5
input_1*
	input_1:0�����������S
global_average_pooling2d7
global_average_pooling2d/Mean:0����������
tensorflow/serving/predict*@
__saved_model_init_op'%
__saved_model_init_op
init_1