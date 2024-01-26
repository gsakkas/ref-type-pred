module Ex7_1 where

import Prelude hiding (map)

{-@ measure size @-}
{-@ size :: [a] -> Nat @-}
size :: [a] -> Int
size [] = 0
size (_:rs) = 1 + size rs

type List a = [a]
{-@ type ListN a N = {v:List a | size v = N} @-}
{-@ type ListX a X = ListN a {size X} @-}

{-@ map :: <mask> @-}
map _ [] = []
map f (x:xs) = f x : map f xs