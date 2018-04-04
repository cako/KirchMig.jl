############
# Utilities 
############
"""
split(N, P, i) -> from:to

Find the ith range when 1:N is split into P consecutive parts of roughly equal size
"""
function split(N, P, i)
    base, rem = divrem(N, P)
    from = (i - 1) * base + min(rem, i - 1) + 1
    to = from + base - 1 + (i ≤ rem ? 1 : 0)
    from : to
end

#########################
# Modeling (demigration)
#########################

# Threaded
kirchmod_thread(model::AbstractArray, t::AbstractVector,
         trav_r::AbstractArray) = kirchmod_thread(model, t, trav_r, trav_r)

function kirchmod_thread(model::AbstractArray{T, N},
                  t::AbstractVector,
                  trav_r::AbstractArray{<:Real, M},
                  trav_s::AbstractArray{<:Real, M}) where {T,N,M}

    Nzxy = size(model)

    nr = size(trav_r)[end]
    ns = size(trav_s)[end]
    nzxy = size(trav_r)[1:M-1]
    NZXY = size(trav_s)[1:M-1]

    ot, dt, nt = t[1], t[2]-t[1], length(t)

    if nzxy != Nzxy
        throw(DimensionMismatch("1st $(M-1) dimensions of trav_r must be the same as that of model"))
    end
    if nzxy != NZXY
        throw(DimensionMismatch("1st $(M-1) dimensions of trav_r and trav_s must be the same"))
    end

    colons = [Colon() for i=1:M-1]

    nthreads = Threads.nthreads()
    v   = Vector{Vector{Matrix{T}}}(nthreads)
    ind = Vector{Vector{Int}}(nthreads)
    Threads.@threads for i in 1:nthreads
        v[i] = Matrix{T}[]
        ind[i] = Int[]
    end
    Threads.@threads for i in 1:nthreads
        P = Threads.threadid()
        range = split(nr, nthreads, P)
        @fastmath @inbounds @simd for ir in range
            rec = zeros(T, ns, nt+1)
            for is=1:ns
                tt = trav_r[colons...,ir] + trav_s[colons...,is]
                its = round.(Int, (tt-ot)/dt)
                its = min.(max.(its, 0), nt)+1
                # Do not remove this loop!
                for ixyz = 1:length(its)
                    rec[is,its[ixyz]] += model[ixyz]
                end
            end
            push!(v[P], rec[:,1:nt])
            push!(ind[P],ir)
        end
    end
    v = vcat(v...)
    ind = vcat(ind...)
    data = zeros(T, nr, ns, nt)
    @inbounds @simd for i=1:nr
        data[ind[i], :, :] = v[i]
    end
    return data
end

# Parallel
kirchmod_par(model::AbstractArray, t::AbstractVector,
         trav_r::AbstractArray) = kirchmod_par(model, t, trav_r, trav_r)
function kirchmod_par(model::AbstractArray{T, N},
                  t::AbstractVector,
                  trav_r::AbstractArray{<:Real, M},
                  trav_s::AbstractArray{<:Real, M}) where {T,N,M}

    Nzxy = size(model)

    nr = size(trav_r)[end]
    ns = size(trav_s)[end]
    nzxy = size(trav_r)[1:M-1]
    NZXY = size(trav_s)[1:M-1]

    ot, dt, nt = t[1], t[2]-t[1], length(t)

    if nzxy != Nzxy
        throw(DimensionMismatch("1st $(M-1) dimensions of trav_r must be the same as that of model"))
    end
    if nzxy != NZXY
        throw(DimensionMismatch("1st $(M-1) dimensions of trav_r and trav_s must be the same"))
    end

    colons = [Colon() for i=1:M-1]
    data = @parallel vcat for ir=1:nr
        rec = zeros(T, ns, nt+1)
        @fastmath @inbounds @simd for is=1:ns
            tt = trav_r[colons...,ir] + trav_s[colons...,is]
            its = round.(Int, (tt-ot)/dt)
            its = min.(max.(its, 0), nt)+1
            # Do not remove this loop!
            for ixyz = 1:length(its)
                rec[is,its[ixyz]] += model[ixyz]
            end
        end
        rec
    end
    return permutedims(reshape(data[:,1:nt], ns, nr, nt), (2, 1, 3))
end

# Serial
kirchmod(model::AbstractArray, t::AbstractVector,
         trav_r::AbstractArray) = kirchmod(model, t, trav_r, trav_r)
function kirchmod(model::AbstractArray{T, N},
                  t::AbstractVector,
                  trav_r::AbstractArray{<:Real, M},
                  trav_s::AbstractArray{<:Real, M}) where {T,N,M}

    Nzxy = size(model)

    nr = size(trav_r)[end]
    ns = size(trav_s)[end]
    nzxy = size(trav_r)[1:M-1]
    NZXY = size(trav_s)[1:M-1]

    ot, dt, nt = t[1], t[2]-t[1], length(t)

    if nzxy != Nzxy
        throw(DimensionMismatch("1st $(M-1) dimensions of trav_r must be the same as that of model"))
    end
    if nzxy != NZXY
        throw(DimensionMismatch("1st $(M-1) dimensions of trav_r and trav_s must be the same"))
    end

    colons = [Colon() for i=1:M-1]
    data = zeros(T, nr, ns, nt+1)
    @fastmath @inbounds for ir=1:nr
        @simd for is=1:ns
            tt = trav_r[colons...,ir] + trav_s[colons...,is]
            its = round.(Int, (tt-ot)/dt)
            its = min.(max.(its, 0), nt)+1
            # Do not remove this loop!
            for ixyz = 1:length(its)
                data[ir,is,its[ixyz]] += model[ixyz]
            end
        end
    end
    return data[:,:,1:nt]
end

##############
# Migration
##############

# Threaded
kirchmig_thread(model::AbstractArray, t::AbstractVector,
         trav_r::AbstractArray) = kirchmig_thread(model, t, trav_r, trav_r)

function kirchmig_thread(data::AbstractArray{T, 3},
                  t::AbstractVector,
                  trav_r::AbstractArray{<:Real, M},
                  trav_s::AbstractArray{<:Real, M}) where {T,M}

    nR = size(trav_r)[end]
    nS = size(trav_s)[end]
    nzxy = size(trav_r)[1:M-1]
    NZXY = size(trav_s)[1:M-1]

    nr, ns, nt = size(data)
    ot, dt, nT = t[1], t[2]-t[1], length(t)

    if nr != nR 
        throw(DimensionMismatch("1st dimension of data (receivers) must be the same as last dimension of trav_r"))
    end
    if ns != nS
        throw(DimensionMismatch("2nd dimension of data (sources) must be the same as last dimension of trav_s"))
    end
    if nzxy != NZXY
        throw(DimensionMismatch("1st $(M-1) dimensions of trav_r and trav_s must be the same"))
    end

    data_ = zeros(T, nr, ns, nt+1)
    data_[:,:,1:nt] = data
    colons = [Colon() for i=1:M-1]

    nthreads = Threads.nthreads()
    v = Vector{Vector{Vector{T}}}(nthreads)
    Threads.@threads for i in 1:nthreads
        v[i] = Vector{T}[]
    end
    Threads.@threads for i in 1:nthreads
        mod = zeros(T, prod(nzxy))
        P = Threads.threadid()
        range = split(nr, nthreads, P)
        @fastmath @inbounds @simd for ir in range
            for is=1:ns
                tt = trav_r[colons..., ir] + trav_s[colons..., is]
                its = round.(Int, (tt-ot)/dt)
                its = min.(max.(its, 0), nt)+1
                mod += data_[ir, is, its[:]]
            end
        end
        push!(v[P], mod)
    end
    return reshape(sum(vcat(v...)), nzxy...)
end

# Parallel
kirchmig_par(model::AbstractArray, t::AbstractVector,
         trav_r::AbstractArray) = kirchmig_par(model, t, trav_r, trav_r)

function kirchmig_par(data::AbstractArray{T, 3},
                  t::AbstractVector,
                  trav_r::AbstractArray{<:Real, M},
                  trav_s::AbstractArray{<:Real, M}) where {T,M}

    nR = size(trav_r)[end]
    nS = size(trav_s)[end]
    nzxy = size(trav_r)[1:M-1]
    NZXY = size(trav_s)[1:M-1]

    nr, ns, nt = size(data)
    ot, dt, nT = t[1], t[2]-t[1], length(t)

    if nr != nR 
        throw(DimensionMismatch("1st dimension of data (receivers) must be the same as last dimension of trav_r"))
    end
    if ns != nS
        throw(DimensionMismatch("2nd dimension of data (sources) must be the same as last dimension of trav_s"))
    end
    if nzxy != NZXY
        throw(DimensionMismatch("1st $(M-1) dimensions of trav_r and trav_s must be the same"))
    end

    data_ = zeros(T, nr, ns, nt+1)
    data_[:,:,1:nt] = data
    colons = [Colon() for i=1:M-1]
    model = @parallel (+) for ir=1:nr
        mod = zeros(T, prod(nzxy))
        @fastmath @inbounds @simd for is=1:ns
            tt = trav_r[colons..., ir] + trav_s[colons..., is]
            its = round.(Int, (tt-ot)/dt)
            its = min.(max.(its, 0), nt)+1
            mod += data_[ir, is, its[:]]
        end
        mod
    end
    return reshape(model, nzxy...)
end

# Serial
kirchmig(model::AbstractArray, t::AbstractVector,
         trav_r::AbstractArray) = kirchmig(model, t, trav_r, trav_r)

function kirchmig(data::AbstractArray{T, 3},
                  t::AbstractVector,
                  trav_r::AbstractArray{<:Real, M},
                  trav_s::AbstractArray{<:Real, M}) where {T,M}

    nR = size(trav_r)[end]
    nS = size(trav_s)[end]
    nzxy = size(trav_r)[1:M-1]
    NZXY = size(trav_s)[1:M-1]

    nr, ns, nt = size(data)
    ot, dt, nT = t[1], t[2]-t[1], length(t)

    if nr != nR 
        throw(DimensionMismatch("1st dimension of data (receivers) must be the same as last dimension of trav_r"))
    end
    if ns != nS
        throw(DimensionMismatch("2nd dimension of data (sources) must be the same as last dimension of trav_s"))
    end
    if nzxy != NZXY
        throw(DimensionMismatch("1st $(M-1) dimensions of trav_r and trav_s must be the same"))
    end

    data_ = zeros(T, nr, ns, nt+1)
    data_[:,:,1:nt] = data
    colons = [Colon() for i=1:M-1]
    model = zeros(T, prod(nzxy))
    @fastmath @inbounds for ir=1:nr
        @simd for is=1:ns
            tt = trav_r[colons..., ir] + trav_s[colons..., is]
            its = round.(Int, (tt-ot)/dt)
            its = min.(max.(its, 0), nt)+1
            model += data_[ir, is, its[:]]
        end
    end
    return reshape(model, nzxy...)
end
