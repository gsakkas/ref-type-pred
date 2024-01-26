module Ex4_8_1 where

import Data.Vector

{-@ type Btwn Lo Hi = {v:Int | Lo <= v && v < Hi} @-}
{-@ type SparseN a N = [(Btwn 0 N, a)] @-}

{-@ sparseProduct :: <mask> @-}
sparseProduct :: Vector Int -> [(Int, Int)] -> Int
sparseProduct x y = go 0 y
    where
        go n []         = n
        go n ((i,v):y') = go (n + (x!i) * v) y'