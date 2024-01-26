module Ex7_2 where

type List a = [a]

{-@ measure size @-}
{-@ size :: [a] -> Nat @-}
size :: [a] -> Int
size [] = 0
size (_:rs) = 1 + size rs

{-@ type ListN a N = {v:List a | size v = N} @-}
{-@ type ListX a X = ListN a {size X} @-}

{-@ reverse' :: xs:List a -> ListX a xs @-}
reverse' xs = rev' [] xs
    where
        {-@ rev' :: <mask> @-}
        rev' :: List a -> List a -> List a
        rev' acc [] = acc
        rev' acc (x:xs) = rev' (x:acc) xs