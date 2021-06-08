using SparsityDetection, SparseArrays
input = rand(nc+nv)
output = rand(nc+nv)
function fc_aug!(output,input)
    output .= fc_aug(input)
end
sparsity_pattern = jacobian_sparsity(fc_aug!,output,input)
jac = Float64.(sparse(sparsity_pattern))
display(spy(jac, marker=2, legend=nothing, c=palette([:black], 2)))

using SparseDiffTools
colors = matrix_colors(jac)

using FiniteDiff
FiniteDiff.finite_difference_jacobian!(jac, fc_aug!, x0, colorvec=colors)

fcalls = 0
function f(dx,x)
  global fcalls += 1
  for i in 2:length(x)-1
    dx[i] = x[i-1] - 2x[i] + x[i+1]
  end
  dx[1] = -2x[1] + x[2]
  dx[end] = x[end-1] - 2x[end]
  nothing
end

using SparsityDetection, SparseArrays
input = rand(10)
output = similar(input)
sparsity_pattern = jacobian_sparsity(f,output,input)
jac = Float64.(sparse(sparsity_pattern))