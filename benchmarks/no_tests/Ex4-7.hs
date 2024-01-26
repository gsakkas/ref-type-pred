module Ex4_7 where

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

{-@ absoluteSum' :: <mask> @-}
absoluteSum' :: Vector Int -> Int
absoluteSum' vec = loop 0 n 0 body
    where
        body i acc = acc + abs (vec ! i)
        n = Data.Vector.length vec