module Ex4_8 where

import Data.Vector

{-@ LIQUID "--no-termination" @-}

{-@ type Btwn Lo Hi  = {v:Int | Lo <= v && v < Hi} @-}
{-@ type Nat         = {v:Int | 0 <= v}            @-}
{-@ type VectorN a N = {v:Vector a | vlen v == N}  @-}

{-@ loop :: lo:Nat -> hi:{Nat|lo <= hi} -> a -> (Btwn lo hi -> a -> a) -> a @-}
loop :: Int -> Int -> a -> (Int -> a -> a) -> a
loop lo hi base f = go base lo
    where
        go acc i
            | i < hi = go (f i acc) (i + 1)
            | otherwise = acc

{-@ dotProduct ::  <mask> @-}
dotProduct :: Vector Int -> Vector Int -> Int
dotProduct x y = loop 0 sz 0 body
    where
        body i acc = acc + (x ! i) * (y ! i)
        sz = Data.Vector.length x