"""
    GBDTs

Grammar-Based Decision Tree (GBDT) is a machine learning model that can be used for the interpretable classification and categorization of heterogeneous multivariate time series data.
"""
module GBDTs

export 
        GBDTNode, 
        GBDT,
        induce_tree, 
        partition,
        members_by_bool,
        classify,
        node_members,
        children_id,
        id,
        label,
        gbes_result,
        isleaf,
        children,
        gini,
        gini_loss

using Discretizers
using Reexport
using StatsBase
using TikzGraphs, LightGraphs
@reexport using AbstractTrees
@reexport using ExprRules
@reexport using ExprOptimization

"""
    GBDTNode

Node object of a GBDT.
"""
struct GBDTNode
    id::Int
    label::Int
    gbes_result::Union{Nothing,ExprOptResult}
    children::Vector{GBDTNode}
end
function GBDTNode(id::Int, label::Int) 
    GBDTNode(id, label, nothing, GBDTNode[])
end

"""
    GBDT

GBDT model produced by induce_tree.
"""
struct GBDT
    tree::GBDTNode
    catdisc::Union{Nothing,CategoricalDiscretizer}
end

"""
    Counter

Mutable int counter
"""
mutable struct Counter
    i::Int
end

"""
    id(node::GBDTNode)

Returns the node id.
"""
id(node::GBDTNode) = node.id
"""
    label(node::GBDTNode)

Returns the node label.
"""
label(node::GBDTNode) = node.label
"""
    gbes_result(node::GBDTNode) 

Returns the result from GBES.
"""
gbes_result(node::GBDTNode) = node.gbes_result
"""
    isleaf(node::GBDTNode) 

Returns true if node is a leaf.
"""
isleaf(node::GBDTNode) = isempty(node.children)

#AbstractTrees interface
AbstractTrees.children(node::GBDTNode) = node.children
function AbstractTrees.printnode(io::IO, node::GBDTNode) 
    print(io, "$(node.id): label=$(node.label)")
    if node.gbes_result != nothing
        r = get(node.gbes_result)
        print(io, ", loss=$(round(r.loss,2)), $(r.expr)")
    end
end

"""
    ishomogeneous(v::AbstractVector{T}) where T

Returns true if all elements in v are the same.
"""
ishomogeneous(v::AbstractVector{T}) where T = length(unique(v)) == 1

"""
    gini_loss(node::RuleNode, grammar::Grammar, X::AbstractVector{T}, y_truth::AbstractVector{Int}, 
                     members::AbstractVector{Int}, eval_module::Module; 
                     w1::Float64=100.0, 
                     w2::Float64=0.1) where T

Default loss function based on gini impurity and number of nodes in the derivation tree.  
See Lee et al. "Interpretable categorization of heterogeneous time series data"
"""
function gini_loss(node::RuleNode, grammar::Grammar, X::AbstractVector{T}, y_truth::AbstractVector{Int}, 
                     members::AbstractVector{Int}, eval_module::Module; 
                     w1::Float64=100.0, 
                     w2::Float64=0.1) where T
    ex = get_executable(node, grammar)
    y_bool = partition(X, members, ex, eval_module)
    members_true, members_false = members_by_bool(members, y_bool)
    return w1*gini(y_truth[members_true], y_truth[members_false]) + w2*length(node)
end

"""
    induce_tree(grammar::Grammar, typ::Symbol, p::ExprOptAlgorithm, X::AbstractVector{XT}, 
                        y::AbstractVector{YT}, max_depth::Int, loss::Function=gini_loss,
                        eval_module::Module=Main) where {XT,YT}

Learn a GBDT from labeled data.  Categorical labels are converted to integers.
# Arguments:
- `grammar::Grammar`: grammar
- `typ::Symbol`: start symbol
- `p::ExprOptAlgorithm`: Parameters for ExprOptimization algorithm
- `X::AbstractVector{XT}`: Input data features, e.g., a MultivariateTimeSeries
- `y::AbstractVector{YT}`: Input (class) labels.
- `max_depth::Int`: Maximum depth of GBDT.
- `loss::Function`: Loss function.  See gini_loss() for function signature.
- `eval_module::Module`: Module in which expressions are evaluated.
"""
function induce_tree(grammar::Grammar, typ::Symbol, p::ExprOptAlgorithm, X::AbstractVector{XT}, 
                        y::AbstractVector{YT}, max_depth::Int, loss::Function=gini_loss,
                        eval_module::Module=Main; kwargs...) where {XT,YT}
    catdisc = CategoricalDiscretizer(y)
    y_truth = encode(catdisc, y)
    induce_tree(grammar, typ, p, X, y_truth, max_depth, loss, eval_module; 
                catdisc=(catdisc), kwargs...)
end
"""
    induce_tree(grammar::Grammar, typ::Symbol, p::ExprOptAlgorithm, X::AbstractVector{T}, 
                        y_truth::AbstractVector{Int}, max_depth::Int, loss::Function=gini_loss,
                        eval_module::Module=Main; 
                        catdisc::Union{Nothing,CategoricalDiscretizer}=nothing),
                        verbose::Bool=false) where T

Learn a GBDT from labeled data.  
# Arguments:
- `grammar::Grammar`: grammar
- `typ::Symbol`: start symbol
- `p::ExprOptAlgorithm`: Parameters for ExprOptimization algorithm
- `X::AbstractVector{XT}`: Input data features, e.g., a MultivariateTimeSeries
- `y_truth::AbstractVector{Int}`: Input (class) labels.
- `max_depth::Int`: Maximum depth of GBDT.
- `loss::Function`: Loss function.  See gini_loss() for function signature.
- `eval_module::Module`: Module in which expressions are evaluated.
- `catdisc::Union{Nothing,CategoricalDiscretizer}`: Discretizer used for converting the labels.
- `min_members_per_branch::Int`: Minimum number of members for a valid branch.
- `prevent_same_label::Bool`: Prevent split if both branches have the same dominant label 
- `verbose::Bool`: Verbose outputs
"""
function induce_tree(grammar::Grammar, typ::Symbol, p::ExprOptAlgorithm, X::AbstractVector{T}, 
                        y_truth::AbstractVector{Int}, max_depth::Int, loss::Function=gini_loss,
                        eval_module::Module=Main; 
                        catdisc::Union{Nothing,CategoricalDiscretizer}=nothing,
                        min_members_per_branch::Int=0,
                        prevent_same_label::Bool=true,
                        verbose::Bool=false) where T
    verbose && println("Starting...")
    @assert length(X) == length(y_truth)
    members = collect(1:length(y_truth))
    node_count = Counter(0)
    node = _split(node_count, grammar, typ, p, X, y_truth, members, max_depth, loss, eval_module,
                 min_members_per_branch=min_members_per_branch, prevent_same_label=prevent_same_label)
    return GBDT(node, catdisc)
end
function _split(node_count::Counter, grammar::Grammar, typ::Symbol, p::ExprOptAlgorithm, 
                       X::AbstractVector{T}, y_truth::AbstractVector{Int}, members::AbstractVector{Int}, 
                       d::Int, loss::Function, eval_module::Module;
                       min_members_per_branch::Int=0,
                       prevent_same_label::Bool=true) where T
    id = node_count.i += 1  #assign ids in preorder
    if d == 0 || ishomogeneous(y_truth[members])
        return GBDTNode(id, mode(y_truth[members]))
    end

    #gbes
    gbes_result = optimize(p, grammar, typ, (node,grammar)->loss(node, grammar, X, y_truth, members, eval_module)) 
    y_bool = partition(X, members, gbes_result.expr, eval_module)
    members_true, members_false = members_by_bool(members, y_bool)

    #don't create split if split doesn't result in two valid groups 
    if length(members_true) <= min_members_per_branch || length(members_false) <= min_members_per_branch
        return GBDTNode(id, mode(y_truth[members]))
    end

    #don't create split if both sides of the split have the same dominant label
    if prevent_same_label && (mode(y_truth[members_true]) == mode(y_truth[members_false]))
        return GBDTNode(id, mode(y_truth[members]))
    end

    child_true = _split(node_count, grammar, typ, p, X, y_truth, members_true, d-1, loss, eval_module;
                       min_members_per_branch=min_members_per_branch,
                       prevent_same_label=prevent_same_label)
    child_false = _split(node_count, grammar, typ, p, X, y_truth, members_false, d-1, loss, eval_module;
                       min_members_per_branch=min_members_per_branch,
                       prevent_same_label=prevent_same_label)

    return GBDTNode(id, mode(y_truth[members]), gbes_result, [child_true, child_false])
end

"""
    partition(X::AbstractVector{T}, members::AbstractVector{Int}, expr, eval_module::Module) where T

Returns a Boolean vector of length members containing the results of evaluating expr on each member.  Expressions are evaluated in eval_module.
"""
function partition(X::AbstractVector{T}, members::AbstractVector{Int}, expr, eval_module::Module) where T
    y_bool = Vector{Bool}(undef, length(members))
    for i in eachindex(members)
        @eval eval_module x = $(X[members[i]])
        y_bool[i] = Core.eval(eval_module, expr) #use x in expression
    end
    y_bool
end

"""
    members_by_bool(members::AbstractVector{Int}, y_bool::AbstractVector{Bool})

Returns a tuple containing the results of splitting members by the Boolean values in y_bool.
"""
function members_by_bool(members::AbstractVector{Int}, y_bool::AbstractVector{Bool})
    @assert length(y_bool) == length(members)
    return members[findall(y_bool)], members[findall(!,y_bool)]
end

"""
    gini(v1::AbstractVector{T}, v2::AbstractVector{T}) where T

Returns the gini impurity of v1 and v2 weighted by number of elements.
"""
function gini(v1::AbstractVector{T}, v2::AbstractVector{T}) where T
    N1, N2 = length(v1), length(v2)
    return (N1*gini(v1) + N2*gini(v2)) / (N1+N2)
end
"""
    gini(v::AbstractVector{T}) where T

Returns the Gini impurity of v.  Returns 0.0 if empty.
"""
function gini(v::AbstractVector{T}) where T
    isempty(v) && return 0.0
    return 1.0 - sum(abs2, proportions(v))
end

"""
    Base.length(model::GBDT)

Returns the number of vertices in the GBDT. 
"""
Base.length(model::GBDT) = length(model.tree)
"""
    Base.length(root::GBDTNode)

Returns the number of vertices in the tree rooted at root.
"""
function Base.length(root::GBDTNode)
    retval = 1
    for c in root.children
        retval += length(c)
    end
    return retval
end

Base.show(io::IO, model::GBDT) = Base.show(io::IO, model.tree)
Base.show(io::IO, tree::GBDTNode) = print_tree(io, tree)

"""
    Base.display(model::GBDT; edgelabels::Bool=false)

Returns a TikzGraphs plot of the tree.  Turn off edgelabels for cleaner plot.  Left branch is true, right branch is false.
"""
function Base.display(model::GBDT; kwargs...)
    display(model.tree, model.catdisc; kwargs...)
end
"""
    Base.display(root::GBDTNode, catdisc::Uniont{Nothing,CategoricalDiscretizer}=nothing;
                     edgelabels::Bool=false)

Returns a TikzGraphs plot of the tree.  Turn off edgelabels for cleaner plot.  Left branch is true, right branch is false.
If catdisc is supplied, use it to decode the labels.
"""
function Base.display(root::GBDTNode, catdisc::Union{Nothing,CategoricalDiscretizer}=nothing;
                     edgelabels::Bool=false)
    n_nodes = length(root)
    g = DiGraph(n_nodes)
    text_labels, edge_labels = Vector{String}(n_nodes), Dict{Tuple{Int,Int},String}() 
    for node in PreOrderDFS(root)
        if node.gbes_result != nothing
            r = node.gbes_result
            text_labels[node.id] = string("$(node.id): $(verbatim(string(r.expr)))")
        else
            label = catdisc != nothing ?  decode(catdisc, node.label) : node.label
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
#Stay in text mode, escape some latex characters
function verbatim(s::String)
    s = replace(s, "_", "\\_")
end

"""
    classify(model::GBDT, X::AbstractVector{T}, members::AbstractVector{Int}, 
                     eval_module::Module=Main) where T

Predict classification label of each member using GBDT model.  Evaluate expressions in eval_module.
"""
function classify(model::GBDT, X::AbstractVector{T}, members::AbstractVector{Int}=collect(1:length(X)), 
                     eval_module::Module=Main) where T
    classify(model.tree, X, members, eval_module; catdisc=model.catdisc)
end
"""
    classify(node::GBDTNode, X::AbstractVector{T}, members::AbstractVector{Int}, eval_module::Module=Main;
                     catdisc::Union{Nothing,CategoricalDiscretizer}=nothing) where T

Predict classification label of each member using GBDT tree.  Evaluate expressions in eval_module.  If catdisc is available, use discretizer to decode labels.
"""
function classify(node::GBDTNode, X::AbstractVector{T}, members::AbstractVector{Int}=collect(1:length(X)), 
                     eval_module::Module=Main; 
                     catdisc::Union{Nothing,CategoricalDiscretizer}=nothing) where T
    y_pred = Vector{Int}(undef,length(members))
    for i in eachindex(members)
        @eval eval_module x=$(X[i])
        y_pred[i] = _classify(node, eval_module) 
    end
    if catdisc == nothing
        return y_pred
    else
        return decode(catdisc, y_pred)
    end
end
function _classify(node::GBDTNode, eval_module::Module)
    isleaf(node) && return node.label

    ex = get_expr(node.gbes_result)
    ch =  Core.eval(eval_module, ex) ? node.children[1] : node.children[2] 
    return _classify(ch, eval_module) 
end

"""
    node_members(model::GBDT, X::AbstractVector{T}, members::AbstractVector{Int}, 
                      eval_module::Module=Main) where T

Returns the members of each node in the tree.
"""
function node_members(model::GBDT, X::AbstractVector{T}, members::AbstractVector{Int}, 
                      eval_module::Module=Main) where T
    node_members(model.tree, X, members, eval_module)
end
"""
    node_members(node::gbdtnode, x::abstractvector{t}, members::abstractvector{int}, 
                      eval_module::module=main) where T

returns the members of each node in the tree.
"""
function node_members(node::GBDTNode, X::AbstractVector{T}, members::AbstractVector{Int}, 
                      eval_module::Module=Main) where T
    mvec = Vector{Vector{Int}}(undef,length(node))
    _node_members!(mvec, node, X, members, eval_module)
    mvec
end
function _node_members!(mvec::Vector{Vector{Int}}, node::GBDTNode, X::AbstractVector{T}, 
                        members::AbstractVector{Int}, eval_module::Module) where T
    mvec[node.id] = deepcopy(members)
    isleaf(node) && return

    ex = get_expr(node.gbes_result)
    y_bool = partition(X, members, ex, eval_module)
    members_true, members_false = members_by_bool(members, y_bool)
    _node_members!(mvec, node.children[1], X, members_true, eval_module)
    _node_members!(mvec, node.children[2], X, members_false, eval_module)
end

"""
    Base.getindex(model::GBDT, id::Int)

returns node with id 
"""
function Base.getindex(model::GBDT, id::Int)
    for node in PreOrderDFS(model.tree)
        if node.id == id
            return node 
        end
    end
    error("node id not found")
end

"""
    children_id(node::GBDTNode) 

returns a vector that contains the node ids of the children of node
"""
children_id(node::GBDTNode) = Int[c.id for c in children(node)]

"""
   get_expr(node::GBDTNode)

returns the expression of the node 
"""
ExprOptimization.get_expr(node::GBDTNode) = get_expr(node.gbes_result)

end # module
