module Ex6_5 where

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

{-@ measure notEmpty @-}
notEmpty :: [a] -> Bool
notEmpty [] = False
notEmpty (_:_) = True

{-@ type NEList a = {v:[a] | notEmpty v} @-}

{-@ risers :: Ord a => NEList a -> NEList [a] @-}
risers :: (Ord a) => [a] -> [[a]]
risers [] = []
risers [x] = [[x]]
risers (x:y:etc)
    | x <= y = (x:s) : ss
    | otherwise = [x] : (s : ss)
        where
            (s, ss) = safeSplit $ risers (y:etc)

{-@ safeSplit :: NEList a -> (a, [a]) @-}
safeSplit (x:xs) = (x, xs)
safeSplit _ = die "don't worry, be happy"

test1 = risers [1,2,3,2,2,3] == [[1,2,3],[2,2,3]]

-- emptyList :: [Int]
-- emptyList = []
-- test2 = risers emptyList == undefined