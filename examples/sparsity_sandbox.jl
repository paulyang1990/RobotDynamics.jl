# pasted from docs
using FiniteDiff, StaticArrays

fcalls = 0
function f(dx,x) # in-place
  global fcalls += 1
  for i in 2:length(x)-1
    dx[i] = x[i-1] - 2x[i] + x[i+1]
  end
  dx[1] = -2x[1] + x[2]
  dx[end] = x[end-1] - 2x[end]
  nothing
end

N = 10
handleleft(x,i) = i==1 ? zero(eltype(x)) : x[i-1]
handleright(x,i) = i==length(x) ? zero(eltype(x)) : x[i+1]
function g(x) # out-of-place
  global fcalls += 1
  @SVector [handleleft(x,i) - 2x[i] + handleright(x,i) for i in 1:N]
end

x = @SVector rand(N)
FiniteDiff.finite_difference_jacobian(g,x)

x = rand(10)
output = zeros(10,10)
@benchmark FiniteDiff.finite_difference_jacobian!($output,$f,$x)
output

cache = FiniteDiff.JacobianCache(x)
@benchmark FiniteDiff.finite_difference_jacobian!($output,$f,$x,$cache) # 0.000008 seconds (7 allocations: 224 bytes)


using SparsityDetection, SparseArrays
input = rand(10)
out = similar(input)
sparsity_pattern = sparsity!(f,out,input)
sparsejac = Float64.(sparse(sparsity_pattern))

using SparseDiffTools
colors = matrix_colors(sparsejac)

sparsecache = FiniteDiff.JacobianCache(x,colorvec=colors,sparsity=sparsejac)
FiniteDiff.finite_difference_jacobian!(sparsejac,f,x,sparsecache)

fcalls = 0
@benchmark FiniteDiff.finite_difference_jacobian!($output,$f,$x,$cache)
fcalls #11

fcalls = 0
@benchmark FiniteDiff.finite_difference_jacobian!($sparsejac,$f,$x,$sparsecache)
fcalls #4