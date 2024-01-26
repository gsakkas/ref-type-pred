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