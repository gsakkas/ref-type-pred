module Ex6_2 where

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

{-@ type NonZero = {v:Int | v /= 0} @-}

{-@ divide :: Int -> NonZero -> Int @-}
divide :: Int -> Int -> Int
divide _ 0 = die "divide-by-zero"
divide x n = x `div` n

{-@ measure notEmpty @-}
notEmpty :: [a] -> Bool
notEmpty []    = False
notEmpty (_:_) = True

{-@ size :: <mask> @-}
size []     = 0
size (_:xs) = 1 + size xs

{-@ type NEList a = {v:[a] | notEmpty v} @-}

{-@ average :: NEList Int -> Int @-}
average xs = divide total elems
    where
        total = sum xs
        elems = size xs