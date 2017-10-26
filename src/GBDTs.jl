module GBDTs

export 
        GBDTNode, 
        GBDTResult,
        induce_tree, 
        partition,
        classify,
        node_members,
        id,
        label,
        gbes_result,
        isleaf,
        children,
        gini,
        gini_loss

using DataFrames
using Discretizers
using Reexport
using StatsBase
using TikzGraphs, LightGraphs
@reexport using AbstractTrees
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
    catdisc::Nullable{CategoricalDiscretizer}
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
    print(io, "$(node.id): label=$(node.label)")
    if !isnull(node.gbes_result)
        r = get(node.gbes_result)
        print(io, ", loss=$(round(r.loss,2)), $(r.expr)")
    end
end

ishomogeneous{T}(v::AbstractVector{T}) = length(unique(v)) == 1

function gini_loss{T}(node::RuleNode, grammar::Grammar, X::AbstractVector{T}, y_truth::AbstractVector{Int}, 
                     members::AbstractVector{Int}, eval_module::Module; 
                     w1::Float64=100.0, 
                     w2::Float64=0.1)
    ex = get_executable(node, grammar)
    y_bool = partition(X, members, ex, eval_module)
    members_true, members_false = members_by_bool(members, y_bool)
    return w1*gini(y_truth[members_true], y_truth[members_false]) + w2*length(node)
end

function induce_tree{XT,YT}(grammar::Grammar, typ::Symbol, p::ExprOptParams, X::AbstractVector{XT}, 
                        y_raw::AbstractVector{YT}, max_depth::Int, loss::Function=gini_loss,
                        eval_module::Module=Main; kwargs...)
    catdisc = CategoricalDiscretizer(y_raw)
    y_truth = encode(catdisc, y_raw)
    induce_tree(grammar, typ, p, X, y_truth, max_depth, loss, eval_module; 
                catdisc=Nullable{CategoricalDiscretizer}(catdisc), kwargs...)
end
function induce_tree{T}(grammar::Grammar, typ::Symbol, p::ExprOptParams, X::AbstractVector{T}, 
                        y_truth::AbstractVector{Int}, max_depth::Int, loss::Function=gini_loss,
                        eval_module::Module=Main; 
                        catdisc::Nullable{CategoricalDiscretizer}=Nullable{CategoricalDiscretizer}(),
                        verbose::Bool=false)
    verbose && println("Starting...")
    @assert length(X) == length(y_truth)
    members = collect(1:length(y_truth))
    node_count = Counter(0)
    node = _split(node_count, grammar, typ, p, X, y_truth, members, max_depth, loss, eval_module)
    return GBDTResult(node, catdisc)
end
function _split{T}(node_count::Counter, grammar::Grammar, typ::Symbol, p::ExprOptParams, 
                       X::AbstractVector{T}, y_truth::AbstractVector{Int}, members::AbstractVector{Int}, 
                       d::Int, loss::Function, eval_module::Module)
    id = node_count.i += 1  #assign ids in preorder
    if d == 0 || ishomogeneous(y_truth[members])
        return GBDTNode(id, mode(y_truth[members]))
    end

    #gbes
    gbes_result = optimize(p, grammar, typ, (node,grammar)->loss(node, grammar, X, y_truth, members, eval_module)) 
    y_bool = partition(X, members, gbes_result.expr, eval_module)
    members_true, members_false = members_by_bool(members, y_bool)

    #don't create split if search was unsuccessful
    if isempty(members_true) || isempty(members_false)
        return GBDTNode(id, mode(y_truth[members]))
    end

    child_true = _split(node_count, grammar, typ, p, X, y_truth, members_true, d-1, loss, eval_module)
    child_false = _split(node_count, grammar, typ, p, X, y_truth, members_false, d-1, loss, eval_module)

    return GBDTNode(id, mode(y_truth[members]), gbes_result, [child_true, child_false])
end

function partition{T}(X::AbstractVector{T}, members::AbstractVector{Int}, expr, eval_module::Module)
    y_bool = Vector{Bool}(length(members))
    for i in eachindex(members)
        @eval eval_module x = $(X[members[i]]::SubDataFrame)
        y_bool[i] = eval(eval_module, expr) #use x in expression
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

"""
    Base.length(root::GBDTNode)

Return the number of vertices in the tree rooted at root.
"""
function Base.length(root::GBDTNode)
    retval = 1
    for c in root.children
        retval += length(c)
    end
    return retval
end

function Base.display(result::GBDTResult; kwargs...)
    display(result.tree, result.catdisc; kwargs...)
end
function Base.display(root::GBDTNode, catdisc::Nullable{CategoricalDiscretizer}=Nullable{CategoricalDiscretizer}();
                     edgelabels::Bool=false)
    n_nodes = length(root)
    g = DiGraph(n_nodes)
    text_labels, edge_labels = Vector{String}(n_nodes), Dict{Tuple{Int,Int},String}() 
    for node in PreOrderDFS(root)
        if !isnull(node.gbes_result)
            r = get(node.gbes_result)
            text_labels[node.id] = string("$(node.id): $(verbatim(string(r.expr)))")
        else
            label = !isnull(catdisc) ?  decode(get(catdisc), node.label) : node.label
            text_labels[node.id] = string("$(node.id): label=$(label)")
        end
        for (i, ch) in enumerate(node.children)
            add_edge!(g, node.id, ch.id)
            edge_labels[(node.id, ch.id)] = i==1 ? "True" : "False"
        end
    end
    if edgelabels
        return TikzGraphs.plot(g, text_labels; edge_labels=edge_labels)
    else
        return TikzGraphs.plot(g, text_labels)
    end
end
function verbatim(s::String)
    s = replace(s, "_", "\\_")
end

function partition{T}(X::AbstractVector{T}, members::AbstractVector{Int}, expr, eval_module::Module)
    y_bool = Vector{Bool}(length(members))
    for i in eachindex(members)
        @eval eval_module x=$(X[members[i]]::SubDataFrame)
        y_bool[i] = eval(eval_module, expr) #use x in expression
    end
    y_bool
end
function classify{T}(result::GBDTResult, X::AbstractVector{T}, members::AbstractVector{Int}, 
                     eval_module::Module=Main)
    classify(result.tree, X, members, eval_module; catdisc=result.catdisc)
end
function classify{T}(node::GBDTNode, X::AbstractVector{T}, members::AbstractVector{Int}, eval_module::Module=Main;
                     catdisc::Nullable{CategoricalDiscretizer}=Nullable{CategoricalDiscretizer}())
    y_pred = Vector{Int}(length(members))
    for i in eachindex(members)
        @eval eval_module x=$(X[i]::SubDataFrame)
        y_pred[i] = _classify(node, eval_module) 
    end
    if isnull(catdisc)
        return y_pred
    else
        return decode(get(catdisc), y_pred)
    end
end
function _classify(node::GBDTNode, eval_module::Module)
    isleaf(node) && return node.label

    ex = get(node.gbes_result).expr
    ch =  eval(eval_module, ex) ? node.children[1] : node.children[2] 
    return _classify(ch, eval_module) 
end

function node_members{T}(result::GBDTResult, X::AbstractVector{T}, members::AbstractVector{Int}, 
                      eval_module::Module=Main)
    node_members(result.tree, X, members, eval_module)
end
function node_members{T}(node::GBDTNode, X::AbstractVector{T}, members::AbstractVector{Int}, 
                      eval_module::Module=Main)
    mvec = Vector{Vector{Int}}(length(node))
    _node_members!(mvec, node, X, members, eval_module)
    mvec
end
function _node_members!{T}(mvec::Vector{Vector{Int}}, node::GBDTNode, X::AbstractVector{T}, 
                        members::AbstractVector{Int}, eval_module::Module)
    mvec[node.id] = deepcopy(members)
    isleaf(node) && return

    ex = get(node.gbes_result).expr
    y_bool = partition(X, members, ex, eval_module)
    members_true, members_false = members_by_bool(members, y_bool)
    _node_members!(mvec, node.children[1], X, members_true, eval_module)
    _node_members!(mvec, node.children[2], X, members_false, eval_module)
end

end # module
