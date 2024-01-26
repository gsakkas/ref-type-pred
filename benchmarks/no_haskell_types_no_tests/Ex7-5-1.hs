module Ex7_5_1 where

import Prelude hiding (fst, snd)

type List a = [a]

{-@ measure size @-}
{-@ size :: List a -> Nat @-}
size :: List a -> Int
size [] = 0
size (_:rs) = 1 + size rs

{-@ measure fst @-}
fst (x, _) = x

{-@ measure snd @-}
snd (_, y) = y

{-@ type ListN a N = {v:List a | size v = N} @-}
{-@ predicate Sum2 X N = size (fst X) + size (snd X) = N @-}

{-@ partition :: <mask> @-}
partition _ []      = ([], [])
partition f (x:xs)
    | f x           = (x:ys, zs)
    | otherwise     = (ys, x:zs)
    where
        (ys,zs)     = partition f xs