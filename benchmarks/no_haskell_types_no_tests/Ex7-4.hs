module Ex7_4 where

import Prelude hiding (drop)

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

type List a = [a]

{-@ measure size @-}
{-@ size :: List a -> Nat @-}
size :: List a -> Int
size [] = 0
size (_:rs) = 1 + size rs

{-@ type ListN a N = {v:List a | size v = N} @-}
{-@ type ListX a X = ListN a {size X} @-}
{-@ type ListGE a N = {v:List a | N <= size v} @-}

{-@ drop :: <mask> @-}
drop 0 xs = xs
drop n (_:xs) = drop (n-1) xs
drop _ _ = die "won't happen"