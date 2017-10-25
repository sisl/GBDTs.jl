module GBDTs

export 
        GBDTNode, 
        induce_tree, 
        classify,
        id,
        label,
        gbes_result,
        isleaf,
        children,
        gini

using AbstractTrees
using Reexport
using StatsBase
@reexport using ExprRules
@reexport using ExprOptimization

struct GBDTNode
    id::Int
    label::Int
    gbes_result::Nullable{ExprOptResult}
    children::Vector{GBDTNode}
end
function GBDTNode(id::Int, label::Int) 
    GBDTNode(id, label, Nullable{ExprOptResult}(), GBDTNode[])
end

struct GBDTResult
    tree::GBDTNode
    #members::Vector{Vector{Int}}
end

mutable struct Counter
    i::Int
end

id(node::GBDTNode) = node.id
label(node::GBDTNode) = node.label
gbes_result(node::GBDTNode) = node.gbes_result
isleaf(node::GBDTNode) = isempty(node.children)

AbstractTrees.children(node::GBDTNode) = node.children
function AbstractTrees.printnode(io::IO, node::GBDTNode) 
    print(io, "$(node.id): label=$(node.label), ")
    if isnull(node.gbes_result)
        print(io, "null")
    else
        r = get(node.gbes_result)
        print(io, "loss=$(r.loss), $(r.expr)")
    end
end

ishomogeneous{T}(v::AbstractVector{T}) = length(unique(v)) == 1

function gini_loss{T}(node::RuleNode, grammar::Grammar, X::AbstractVector{T}, y_truth::AbstractVector{Int}, 
                     members::AbstractVector{Int}; 
                     w1::Float64=100.0, w2::Float64=0.1)
    ex = get_executable(node, grammar)
    @show ex
    @show @time y_bool = classify(X, members, ex)
    members_true, members_false = members_by_bool(members, y_bool)
    return w1*gini(y_truth[members_true], y_truth[members_false]) + w2*length(node)
end

function induce_tree{T}(grammar::Grammar, typ::Symbol, p::ExprOptParams, X::AbstractVector{T}, 
                        y_truth::AbstractVector{Int}, max_depth::Int, loss::Function=gini_loss; 
                        verbose::Bool=true)
    verbose && println("Starting...")
    @assert length(X) == length(y_truth)
    members = collect(1:length(y_truth))
    node_count = Counter(0)
    node = _split(node_count, grammar, typ, p, X, y_truth, members, max_depth, loss)
    return GBDTResult(node)
end
function _split{T}(node_count::Counter, grammar::Grammar, typ::Symbol, p::ExprOptParams, 
                       X::AbstractVector{T}, y_truth::AbstractVector{Int}, members::AbstractVector{Int}, 
                       d::Int, loss::Function)
    if d == 0 || ishomogeneous(y_truth[members])
        return GBDTNode(node_count.i+=1, mode(y_truth[members]))
    end
    id = node_count.i += 1
    gbes_result = optimize(p, grammar, typ, (node,grammar)->loss(node, grammar, X, y_truth, members)) 
    y_bool = classify(X, members, gbes_result.expr)
    members_true, members_false = members_by_bool(members, y_bool)
    child_true = _split(node_count, grammar, typ, p, X, y_truth, members_true, d-1, loss)
    child_false = _split(node_count, grammar, typ, p, X, y_truth, members_false, d-1, loss)

    return GBDTNode(id, mode(y_truth[members]), gbes_result, [child_true, child_false])
end

function classify{T}(X::AbstractVector{T}, members::AbstractVector{Int}, expr)
    y_bool = Vector{Bool}(length(members))
    for i in eachindex(members)
        global x = X[members[i]]::SubDataFrame
        y_bool[i] = eval(expr) #use x in expression
    end
    y_bool
end

function members_by_bool(members::AbstractVector{Int}, y_bool::AbstractVector{Bool})
    @assert length(y_bool) == length(members)
    return members[find(y_bool)], members[find(!,y_bool)]
end

function gini{T}(v1::AbstractVector{T}, v2::AbstractVector{T})
    N1, N2 = length(v1), length(v2)
    return (N1*gini(v1) + N2*gini(v2)) / (N1+N2)
end
function gini{T}(v::AbstractVector{T})
    isempty(v) && return 0.0
    return 1.0 - sum(abs2, proportions(v))
end

end # module
