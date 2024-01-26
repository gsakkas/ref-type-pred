module Ex7_1 where

import Prelude hiding (map)

{-@ type TRUE = {v:Bool | v } @-}
{-@ type FALSE = {v:Bool | not v} @-}

{-@ measure size @-}
{-@ size :: [a] -> Nat @-}
size :: [a] -> Int
size [] = 0
size (_:rs) = 1 + size rs

type List a = [a]
{-@ type ListN a N = {v:List a | size v = N} @-}
{-@ type ListX a X = ListN a {size X} @-}

{-@ map :: <mask> @-}
map :: (a -> b) -> [a] -> [b]
map _ [] = []
map f (x:xs) = f x : map f xs

{-@ prop_map :: List a -> TRUE @-}
prop_map xs = size ys == size xs
    where
        ys = map id xs

test1 = map (*2) [0, 1, 3] == [0, 2, 6]
test2 = map (==1) [0, 1, 3] == [False, True, False]