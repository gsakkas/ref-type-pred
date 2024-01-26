module Ex5_0_1 where

import Data.Vector

{-@ type Nat         = {v:Int | 0 <= v}            @-}
{-@ type Btwn Lo Hi  = {v:Int | Lo <= v && v < Hi} @-}
{-@ type SparseN a N = {v:Sparse a | spDim v == N} @-}

{-@ data Sparse a = SP { spDim   :: Nat
                       , spElems :: [(Btwn 0 spDim, a)]} @-}
data Sparse a = SP { spDim   :: Int
                   , spElems :: [(Int, a)] }

{-@ dotProd :: <mask> @-}
dotProd x (SP _ y) = go 0 y
    where
        go sum ((i, v) : y') = go (sum + (x ! i) * v) y'
        go sum []            = sum