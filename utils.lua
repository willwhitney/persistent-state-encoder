local T = require 'pl.tablex'

-- From https://gist.github.com/cwarden/1207556
function catch(what)
   return what[1]
end

-- From https://gist.github.com/cwarden/1207556
function try(what)
   status, result = pcall(what[1])
   if not status then
      what[2](result)
   end
   return result
end

function subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

-- merge t2 into t1
function merge_tables(t1, t2)
    -- Merges t2 and t1, overwriting t1 keys by t2 keys when applicable
    merged_table = T.deepcopy(t1)
    for k,v in pairs(t2) do
        -- if merged_table[k] then
        --     error('t1 and t2 both contain the key: ' .. k)
        -- end
        merged_table[k]  = v
    end
    return merged_table
end

-- merge t2 into t1
-- TODO do set functions
function merge_tables_by_value(t1, t2)
    -- Merges t2 and t1, overwriting t1 keys by t2 keys when applicable
    for k,v in pairs(t1) do assert(type(k) == 'number') end
    merged_table = T.deepcopy(t1)
    for _,v in pairs(t2) do
        if not isin(v, merged_table) then
            merged_table[#merged_table+1] = v  -- just append
        end
    end
    return merged_table
end

function intersect(t1, t2)
    local intersect_table = {}
    for k,v1 in pairs(t1) do
        if isin(v1, t2) then
            intersect_table[#intersect_table+1] = v1
        end
    end
    return intersect_table
end

function is_subset(small_table, big_table)
    for _, el in pairs(small_table) do
        if not isin(el, big_table) then
            return false
        end
    end
    return true
end

function isin(element, table)
    for _,v in pairs(table) do
        if v == element then
            return true
        end
    end
    return false
end

function is_empty(table)
    if next(table) == nil then return true end
end

-- BUG! If the arg is nil, then it won't get passed into args_table!
function all_args_exist(args_table, num_args)
    if not(#args_table == num_args) then return false end
    local exist = true
    for _,a in pairs(args_table) do
        if a == nil then
            exist = false
        end
    end
    -- assert(false)
    return exist
end

function is_substring(substring, string)
    return not (string:find(substring) == nil)
end

function notnil(x)
    return not(x == nil)
end

-- from http://lua-users.org/wiki/FunctionalLibrary
-- map(function, table)
-- e.g: map(double, {1,2,3})    -> {2,4,6}
function map(func, tbl)
    local newtbl = {}
    for i,v in pairs(tbl) do
        newtbl[i] = func(v)
    end
    return newtbl
end

-- from http://lua-users.org/wiki/FunctionalLibrary
-- filter(function, table)
-- e.g: filter(is_even, {1,2,3,4}) -> {2,4}
function filter(func, tbl)
    local newtbl= {}
    for i,v in pairs(tbl) do
        if func(v) then
        newtbl[i]=v
        end
    end
    return newtbl
end

-- from http://lua-users.org/wiki/FunctionalLibrary
-- head(table)
-- e.g: head({1,2,3}) -> 1
function head(tbl)
    return tbl[1]
end

-- from http://lua-users.org/wiki/FunctionalLibrary
-- tail(table)
-- e.g: tail({1,2,3}) -> {2,3}
--
-- XXX This is a BAD and ugly implementation.
-- should return the address to next porinter, like in C (arr+1)
function tail(tbl)
    if table.getn(tbl) < 1 then
        return nil
    else
        local newtbl = {}
        local tblsize = table.getn(tbl)
        local i = 2
        while (i <= tblsize) do
            table.insert(newtbl, i-1, tbl[i])
            i = i + 1
        end
       return newtbl
    end
end

-- from http://lua-users.org/wiki/FunctionalLibrary
-- foldr(function, default_value, table)
-- e.g: foldr(operator.mul, 1, {1,2,3,4,5}) -> 120
function foldr(func, val, tbl)
    for i,v in pairs(tbl) do
        val = func(val, v)
    end
    return val
end

-- from http://lua-users.org/wiki/FunctionalLibrary
-- reduce(function, table)
-- e.g: reduce(operator.add, {1,2,3,4}) -> 10
function reduce(func, tbl)
    return foldr(func, head(tbl), tail(tbl))
end

-- range(start)             returns an iterator from 1 to a (step = 1)
-- range(start, stop)       returns an iterator from a to b (step = 1)
-- range(start, stop, step) returns an iterator from a to b, counting by step.
-- from http://lua-users.org/wiki/RangeIterator
function range (i, to, inc)
     if i == nil then return end -- range(--[[ no args ]]) -> return "nothing" to fail the loop in the caller

    if not to then
        to = i
        i  = to == 0 and 0 or (to > 0 and 1 or -1)
    end

    -- we don't have to do the to == 0 check
    -- 0 -> 0 with any inc would never iterate
    inc = inc or (i < to and 1 or -1)

    -- step back (once) before we start
    i = i - inc

    return function () if i == to then return nil end i = i + inc return i, i end
end

-- print(merge_tables_by_value({['a']=1}, {['b'] = 2, ['c'] = 5}))

-- print(intersect({'a','b','c'}, {'d','b','c'}))
