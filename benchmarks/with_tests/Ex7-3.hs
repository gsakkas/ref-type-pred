module Ex7_3 where

import Prelude hiding (zipWith)

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

type List a = [a]

{-@ type ListN a N = {v:List a | len v = N} @-}
{-@ type ListX a X = ListN a {len X} @-}

{-@ measure notEmpty @-}
notEmpty       :: [a] -> Bool
notEmpty []    = False
notEmpty (_:_) = True

{-@ zipWith :: (a -> b -> c) -> xs:List a -> ListX b xs -> ListX c xs @-}
zipWith :: (a -> b -> c) -> List a -> List b -> List c
zipWith f (a:as) (b:bs) = f a b : zipWith f as bs
zipWith _ [] [] = []
zipWith _ _ _ = die "no other cases"

{-@ zipOrNull :: <mask> @-}
zipOrNull :: [a] -> [b] -> [(a, b)]
zipOrNull [] _ = []
zipOrNull _ [] = []
zipOrNull xs ys = zipWith (,) xs ys

{-@ test1 :: {v: _ | len v = 2} @-}
test1 :: [(Int, Bool)]
test1 = zipOrNull [0, 1] [True, False]
test1' = test1 == [(0, True), (1, False)]

{-@ test2 :: {v: _ | len v = 0} @-}
test2 :: [(Int, Bool)]
test2 = zipOrNull [] [True, False]
test2' = test2 == []

{-@ test3 :: {v: _ | len v = 0} @-}
test3 :: [(String, Int)]
test3 = zipOrNull ["cat", "dog"] []
test3' = test3 == []