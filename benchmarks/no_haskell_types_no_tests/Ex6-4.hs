module Ex6_4 where

import Prelude hiding (map)

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

{-@ type Pos = {v:Int | 0 < v} @-}
{-@ type Zero = {v:Int | v == 0} @-}
{-@ type NonZero = {v:Int | v /= 0} @-}

{-@ measure notEmpty @-}
notEmpty :: [a] -> Bool
notEmpty [] = False
notEmpty (_:_) = True

{-@ type NEList a = {v:[a] | notEmpty v} @-}

{-@ divide :: Int -> NonZero -> Int @-}
divide :: Int -> Int -> Int
divide _ 0 = die "divide-by-zero"
divide x n = x `div` n

map :: (a -> b) -> [a] -> [b]
map _ [] = []
map f (x:xs) = f x : map f xs

{-@ wtAverage :: <mask> @-}
wtAverage wxs = divide totElems totWeight
    where
        elems     = map (\(w, x) -> w * x) wxs
        weights   = map (\(w, _) -> w ) wxs
        totElems  = sum elems
        totWeight = sum weights
        sum       = foldl1 (+)