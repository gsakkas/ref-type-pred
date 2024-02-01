module Ex7_5 where

import Prelude hiding (take)

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

{-@ take :: n:Nat -> xs:List a -> {v:List a | if (n <= size xs) then (size v) = n else (size v) = (size xs)} @-}
take :: Int -> List a -> List a
take 0 _      = []
take _ []     = []
take n (x:xs) = x : take (n-1) xs

{-@ test1 :: ListN String 2 @-}
test1 = take 2 ["cat", "dog", "mouse"]
test1' = test1 == ["cat", "dog"]

{-@ test2 :: ListN String 2 @-}
test2 = take 20 ["cow", "goat"]
test2' = test2 == ["mouse"]