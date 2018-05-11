using GBDTs
using Base.Test

let
    X = [true, true, true, false, false] 
    y = [x ? 1 : 2 for x in X]

    grammar = @grammar begin
        b = x | true | false
    end

    srand(0)
    p = MonteCarlo(10,5)
    members = collect(1:length(X))
    model = induce_tree(grammar, :b, p, X, y, 2)
    @test length(model) == 3
    @test id(model.tree) == 1
    @test label(model.tree) == 1
    @test !isleaf(model.tree)
    leafs = collect(Leaves(model.tree))
    @test length(leafs) == 2
    @test sort([n.label for n in leafs]) == [1, 2]
    y_bool = partition(X, members, :x, Main)
    @test X == y_bool
    members_true, members_false = members_by_bool(members, y_bool) 
    @test members_true == find(X) 
    @test members_false == find(!, X) 
    y_pred = classify(model, X)
    @test y_pred == y
    mvec = node_members(model, X, members)
    @test mvec[1] == members
    @test mvec[2] == members_true
    @test mvec[3] == members_false
    @test model.tree == model[1]
    @test children_id(model.tree) == [2,3]
    @test get_expr(nothing) == nothing
    @test get_expr(model[1]) == :x
end
