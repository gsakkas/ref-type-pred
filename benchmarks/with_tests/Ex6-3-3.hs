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

{-@ test1 :: [NEList Int] @-}
test1 :: [[Int]]
test1 = groupEq [0, 1, 2, 2, 1, 3, 3, 3]
test1' = test1 == [[0],[1],[2,2]]

{-@ test2 :: [NEList Int] @-}
test2 :: [[Int]]
test2 = groupEq [0, 1, 2, 2, 1, 3, 3, 3]
test2' = test2 == [[0],[1],[2,2],[1],[3,3,3]]
-- test3 = groupEq $ "ssstringssss liiiiiike thisss"
-- test3' = map head test3 == "strings like this"