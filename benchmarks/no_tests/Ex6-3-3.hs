module Ex6_3_3 where

{-@ measure notEmpty @-}
notEmpty :: [a] -> Bool
notEmpty [] = False
notEmpty (_:_) = True

{-@ type NEList a = {v:[a] | notEmpty v} @-}

{-@ groupEq :: <mask> @-}
groupEq :: (Eq a) => [a] -> [[a]]
groupEq []     = []
groupEq (x:xs) = (x:ys) : groupEq zs
    where
        (ys, zs) = span (x ==) xs