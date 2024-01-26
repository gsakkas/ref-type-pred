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
drop :: Int -> List a -> List a
drop 0 xs = xs
drop n (_:xs) = drop (n-1) xs
drop _ _ = die "won't happen"

{-@ test1 :: ListN String 2 @-}
test1 :: List String
test1 = drop 1 ["cat", "dog", "mouse"]
test1' = test1 == ["dog", "mouse"]

{-@ test2 :: ListN Int 3 @-}
test2 :: List Int
test2 = drop 2 [1, 4, 2, 4, 6]
test2' = test2 == [2, 4, 6]

-- {-@ test3 :: {v:_ | false} @-}
-- test3 :: List Int
-- test3 = drop 1 []
-- test3' = test3 == undefined

-- {-@ test4 :: {v:_ | false} @-}
-- test4 :: List Int
-- test4 = drop (-1) [2, 3, 1]
-- test4' = test4 == undefined