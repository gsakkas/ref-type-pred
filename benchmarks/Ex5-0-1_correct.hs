module Ex5_0_1 where

import Data.Vector

{-@ type Nat         = {v:Int | 0 <= v}            @-}
{-@ type Btwn Lo Hi  = {v:Int | Lo <= v && v < Hi} @-}
{-@ type SparseN a N = {v:Sparse a | spDim v == N} @-}

{-@ data Sparse a = SP { spDim   :: Nat
                       , spElems :: [(Btwn 0 spDim, a)]} @-}
data Sparse a = SP { spDim   :: Int
                   , spElems :: [(Int, a)] }

{-@ dotProd :: x:Vector Int -> SparseN Int (vlen x) -> Int @-}
dotProd :: Vector Int -> Sparse Int -> Int
dotProd x (SP _ y) = go 0 y
    where
        go sum ((i, v) : y') = go (sum + (x ! i) * v) y'
        go sum []            = sum

test1 = dotProd (fromList [1,2,3,4]) (SP 4 [(0, 4), (3, 1)]) == 8
-- test2 = dotProd (fromList [1,2,3]) (SP 3 [(0, 4), (3, 1)]) == undefined