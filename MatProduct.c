#include <petsc.h>

#if defined(PETSC_HAVE_MKL)
#include <mkl.h>
#define PetscStackCallMKLSparse(func,args) do {                                                           \
    sparse_status_t __ierr;                                                                               \
    PetscStackPush(#func);                                                                                \
    __ierr = func args;                                                                                   \
    PetscStackPop;                                                                                        \
    if (__ierr != SPARSE_STATUS_SUCCESS) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in %s(): error code %d",#func,(int)__ierr); \
  } while (0)
#else
#define PetscStackCallMKLSparse(func,args) do {                                                           \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No MKL support"); \
  } while (0)
#endif

int main(int argc, char** argv) {
  Mat            A, C, D, Ct = NULL;
  PetscInt       nbs = 10, ntype = 10, nN = 8, m, M, trial = 5;
  PetscViewer    viewer;
  PetscInt       bs[10], N[8];
  char           *type[10];
  PetscMPIInt    size;
  PetscBool      flg, iscuda, check = PETSC_FALSE, trans = PETSC_FALSE, mkl;
  char           file[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(ierr);
  if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only");
  ierr = PetscOptionsGetString(NULL, NULL, "-f", file, PETSC_MAX_PATH_LEN, &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate binary file with the -f option");
  ierr = PetscOptionsGetInt(NULL, NULL, "-trial", &trial, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(NULL, NULL, "-bs", bs, &nbs, &flg);CHKERRQ(ierr);
  if (!flg) {
    nbs = 1;
    bs[0] = 1;
  }
  ierr = PetscOptionsGetIntArray(NULL, NULL, "-N", N, &nN, &flg);CHKERRQ(ierr);
  if (!flg) {
    nN = 8;
    N[0] = 1;  N[1] = 2;  N[2] = 4;  N[3] = 8;
    N[4] = 16; N[5] = 32; N[6] = 64; N[7] = 128;
  }
  ierr = PetscOptionsGetStringArray(NULL, NULL, "-type", type, &ntype, &flg);CHKERRQ(ierr);
  if (!flg) {
    ntype = 1;
    ierr = PetscStrallocpy(MATSEQAIJ,&type[0]);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetBool(NULL, NULL, "-check_results", &check, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-test_trans", &trans, NULL);CHKERRQ(ierr);
  for (int j = 0; j < nbs; ++j) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatLoad(A, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJ, MATMPIAIJ, "");CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate a MatAIJ input matrix");
    ierr = MatGetSize(A, &m, &M);CHKERRQ(ierr);
    if (m == M) {
      Mat oA;
      ierr = MatTranspose(A, MAT_INITIAL_MATRIX, &oA);CHKERRQ(ierr);
      ierr = MatAXPY(A, 1.0, oA, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDestroy(&oA);CHKERRQ(ierr);
      ierr = MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = MatGetLocalSize(A, &m, NULL);CHKERRQ(ierr);
    ierr = MatGetSize(A, &M, NULL);CHKERRQ(ierr);
    if (bs[j] > 1) {
      Mat               T, Tt, B;
      const PetscScalar *ptr;
      PetscScalar       *val, *Aa;
      const PetscInt    *Ai, *Aj;
      PetscInt          An, i, k;
      PetscBool         done;

      ierr = MatCreateDense(PETSC_COMM_SELF, bs[j], bs[j], bs[j], bs[j], NULL, &T);CHKERRQ(ierr);
      ierr = MatSetRandom(T, NULL);CHKERRQ(ierr);
      ierr = MatTranspose(T, MAT_INITIAL_MATRIX, &Tt);CHKERRQ(ierr);
      ierr = MatAXPY(T, 1.0, Tt, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDestroy(&Tt);CHKERRQ(ierr);
      ierr = MatDenseGetArrayRead(T, &ptr);CHKERRQ(ierr);
      ierr = MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &An, &Ai, &Aj, &done);CHKERRQ(ierr);
      if (!done || An != m) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Inconsistent sizes");
      ierr = MatSeqAIJGetArray(A, &Aa);CHKERRQ(ierr);
      ierr = MatCreate(PETSC_COMM_WORLD, &B);CHKERRQ(ierr);
      ierr = MatSetType(B, MATSEQBAIJ);CHKERRQ(ierr);
      ierr = MatSetSizes(B, bs[j] * An, bs[j] * An, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
      ierr = PetscMalloc1(Ai[An] * bs[j] * bs[j],&val);CHKERRQ(ierr);
      for(i = 0; i < Ai[An]; ++i)
        for(k = 0; k < bs[j] * bs[j]; ++k)
          val[i * bs[j] * bs[j] + k] = Aa[i] * ptr[k];
      ierr = MatSetOption(B, MAT_ROW_ORIENTED, PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatSeqBAIJSetPreallocationCSR(B, bs[j], Ai, Aj, val);CHKERRQ(ierr);
      ierr = PetscFree(val);CHKERRQ(ierr);
      ierr = MatSeqAIJRestoreArray(A, &Aa);CHKERRQ(ierr);
      ierr = MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &An, &Ai, &Aj, &done);CHKERRQ(ierr);
      ierr = MatDenseRestoreArrayRead(T, &ptr);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
      ierr = MatDestroy(&A);CHKERRQ(ierr);
      A    = B;
      /* reconvert back to SeqAIJ before converting to the desired type later */
      ierr = MatConvert(A, MATSEQAIJ, MAT_INPLACE_MATRIX, &A);CHKERRQ(ierr);
      ierr = MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
    }
    for (int i = 0; i < ntype; ++i) {
      char        *tmp;
      PetscInt    *ia_ptr, *ja_ptr;
      PetscScalar *a_ptr;
#if defined(PETSC_HAVE_MKL)
      struct matrix_descr descr;
      sparse_matrix_t     spr;
      descr.type = SPARSE_MATRIX_TYPE_GENERAL;
      descr.diag = SPARSE_DIAG_NON_UNIT;
#endif

      ierr = PetscStrstr(type[i],"mkl",&tmp);CHKERRQ(ierr);
      if (tmp) {
        size_t mlen,tlen;
        char base[256];

        mkl  = PETSC_TRUE;
        ierr = PetscStrlen(tmp, &mlen);CHKERRQ(ierr);
        ierr = PetscStrlen(type[i], &tlen);CHKERRQ(ierr);
        ierr = PetscStrncpy(base, type[i], tlen-mlen+1);CHKERRQ(ierr);
        ierr = MatConvert(A, base, MAT_INPLACE_MATRIX, &A);CHKERRQ(ierr);
      } else {
        mkl  = PETSC_FALSE;
        ierr = MatConvert(A, type[i], MAT_INPLACE_MATRIX, &A);CHKERRQ(ierr);
      }
      ierr = PetscObjectTypeCompareAny((PetscObject)A,&iscuda,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE,"");CHKERRQ(ierr);
      if (mkl) {
        const PetscInt *Ai, *Aj;
        PetscInt       An,ii;
        PetscBool      done;

        ierr = PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJ, MATSEQBAIJ, MATSEQSBAIJ, "");CHKERRQ(ierr);
        if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Not implemented");
        ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &flg);CHKERRQ(ierr);
        ierr = MatGetRowIJ(A, 0, PETSC_FALSE, flg ? PETSC_FALSE : PETSC_TRUE, &An, &Ai, &Aj, &done);CHKERRQ(ierr);
        if (!done) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Inconsistent sizes");
        ierr = PetscMalloc1(An+1,&ia_ptr);CHKERRQ(ierr);
        ierr = PetscMalloc1(Ai[An],&ja_ptr);CHKERRQ(ierr);
        if (flg) { /* SeqAIJ */
          for (ii = 0; ii < An+1; ii++) ia_ptr[ii] = Ai[ii];
          for (ii = 0; ii < Ai[An]; ii++) ja_ptr[ii] = Aj[ii];
          ierr = MatSeqAIJGetArray(A, &a_ptr);CHKERRQ(ierr);
          PetscStackCallMKLSparse(mkl_sparse_d_create_csr,(&spr, SPARSE_INDEX_BASE_ZERO, An, An, ia_ptr, ia_ptr + 1, ja_ptr, a_ptr));
        } else {
          ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQBAIJ, &flg);CHKERRQ(ierr);
          if (flg) {
            for (ii = 0; ii < An+1; ii++) ia_ptr[ii] = Ai[ii] + 1; /* Use fortran indexing to maximize cases covered by _mm routines */
            for (ii = 0; ii < Ai[An]; ii++) ja_ptr[ii] = Aj[ii] + 1; /* Use fortran indexing to maximize cases covered by _mm routines */
            ierr = MatSeqBAIJGetArray(A, &a_ptr);CHKERRQ(ierr);
            PetscStackCallMKLSparse(mkl_sparse_d_create_bsr,(&spr, SPARSE_INDEX_BASE_ONE, SPARSE_LAYOUT_COLUMN_MAJOR, An, An, bs[j], ia_ptr, ia_ptr + 1, ja_ptr, a_ptr));
          } else {
            ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQSBAIJ, &flg);CHKERRQ(ierr);
            if (flg) {
              for (ii = 0; ii < An+1; ii++) ia_ptr[ii] = Ai[ii] + 1; /* Use fortran indexing to maximize cases covered by _mm routines */
              for (ii = 0; ii < Ai[An]; ii++) ja_ptr[ii] = Aj[ii] + 1; /* Use fortran indexing to maximize cases covered by _mm routines */
              ierr = MatSeqSBAIJGetArray(A, &a_ptr);CHKERRQ(ierr);
              PetscStackCallMKLSparse(mkl_sparse_d_create_bsr,(&spr, SPARSE_INDEX_BASE_ONE, SPARSE_LAYOUT_COLUMN_MAJOR, An, An, bs[j], ia_ptr, ia_ptr + 1, ja_ptr, a_ptr));
#if defined(PETSC_HAVE_MKL)
              descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
              descr.mode = SPARSE_FILL_MODE_UPPER;
              descr.diag = SPARSE_DIAG_NON_UNIT;
#endif
            }
          }
        }
        ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &flg);CHKERRQ(ierr);
        ierr = MatRestoreRowIJ(A, 0, PETSC_FALSE, flg ? PETSC_FALSE : PETSC_TRUE, &An, &Ai, &Aj, &done);CHKERRQ(ierr);
      }

      ierr = MatViewFromOptions(A, NULL, "-A_view");CHKERRQ(ierr);

      for(int k = 0; k < nN; ++k) {
        MatType       Atype,Ctype;
        PetscInt      AM,AN,CM,CN,mm;
        PetscLogStage stage,tstage;
        char          stage_s[256];

        ierr = MatCreateDense(PETSC_COMM_WORLD, bs[j] * m, PETSC_DECIDE, bs[j] * M, N[k], NULL, &C);CHKERRQ(ierr);
        ierr = MatCreateDense(PETSC_COMM_WORLD, bs[j] * m, PETSC_DECIDE, bs[j] * M, N[k], NULL, &D);CHKERRQ(ierr);
        ierr = MatSetRandom(C, NULL);CHKERRQ(ierr);
        if (iscuda) { /* convert to GPU if needed */
          ierr = MatConvert(C, MATDENSECUDA, MAT_INPLACE_MATRIX, &C);CHKERRQ(ierr);
          ierr = MatConvert(D, MATDENSECUDA, MAT_INPLACE_MATRIX, &D);CHKERRQ(ierr);
        }
        if (mkl) {
          if (N[k] > 1) PetscStackCallMKLSparse(mkl_sparse_set_mm_hint,(spr,SPARSE_OPERATION_NON_TRANSPOSE, descr, SPARSE_LAYOUT_COLUMN_MAJOR, N[k], 1 + trial));
          else          PetscStackCallMKLSparse(mkl_sparse_set_mv_hint,(spr,SPARSE_OPERATION_NON_TRANSPOSE, descr, 1 + trial));
          PetscStackCallMKLSparse(mkl_sparse_set_memory_hint,(spr, SPARSE_MEMORY_AGGRESSIVE));
          PetscStackCallMKLSparse(mkl_sparse_optimize,(spr));
        }
        ierr = MatGetType(A,&Atype);CHKERRQ(ierr);
        ierr = MatGetType(C,&Ctype);CHKERRQ(ierr);
        ierr = MatGetSize(A,&AM,&AN);CHKERRQ(ierr);
        ierr = MatGetSize(C,&CM,&CN);CHKERRQ(ierr);

        ierr = PetscSNPrintf(stage_s,sizeof(stage_s),"notrans_type_%s-bs_%D-N_%02d",type[i],bs[j],(int)N[k]);CHKERRQ(ierr);
        ierr = PetscLogStageRegister(stage_s, &stage);CHKERRQ(ierr);
        if (trans && N[k] > 1) {
          ierr = PetscSNPrintf(stage_s,sizeof(stage_s),"trans_type_%s-bs_%D-N_%02d",type[i],bs[j],(int)N[k]);CHKERRQ(ierr);
          ierr = PetscLogStageRegister(stage_s, &tstage);CHKERRQ(ierr);
        }

        /* A*B */
        if (N[k] > 1) {
          ierr = MatProductCreateWithMat(A, C, NULL, D);CHKERRQ(ierr);
          ierr = MatProductSetType(D, MATPRODUCT_AB);CHKERRQ(ierr);
          ierr = MatProductSetFromOptions(D);CHKERRQ(ierr);
          ierr = MatProductSymbolic(D);CHKERRQ(ierr);

          if (!mkl) {
            ierr = MatProductNumeric(D);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD,"Benchmarking MatProduct %s: with A %s %Dx%D and B %s %Dx%D\n",MatProductTypes[MATPRODUCT_AB],Atype,AM,AN,Ctype,CM,CN);
            ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
            for (mm = 0; mm < trial; ++mm) {
              ierr = MatProductNumeric(D);CHKERRQ(ierr);
            }
            ierr = PetscLogStagePop();CHKERRQ(ierr);
          } else {
            const PetscScalar *c_ptr;
            PetscScalar       *d_ptr;

            ierr = MatDenseGetArrayRead(C, &c_ptr);CHKERRQ(ierr);
            ierr = MatDenseGetArrayWrite(D, &d_ptr);CHKERRQ(ierr);
            PetscStackCallMKLSparse(mkl_sparse_d_mm,(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, SPARSE_LAYOUT_COLUMN_MAJOR, c_ptr, CN, CM, 0.0, d_ptr, CM));
            ierr = PetscPrintf(PETSC_COMM_WORLD,"Benchmarking mkl_sparse_d_mm (COLUMN_MAJOR): with A %s %Dx%D and B %s %Dx%D\n",Atype,AM,AN,Ctype,CM,CN);
            ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
            for (mm = 0; mm < trial; ++mm) {
              PetscStackCallMKLSparse(mkl_sparse_d_mm,(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, SPARSE_LAYOUT_COLUMN_MAJOR, c_ptr, CN, CM, 0.0, d_ptr, CM));
            }
            ierr = PetscLogStagePop();CHKERRQ(ierr);
            ierr = MatDenseRestoreArrayWrite(D, &d_ptr);CHKERRQ(ierr);
            ierr = MatDenseRestoreArrayRead(C, &c_ptr);CHKERRQ(ierr);
          }
        } else if (!mkl) {
          Vec cC, cD;

          ierr = MatDenseGetColumnVecRead(C, 0, &cC);CHKERRQ(ierr);
          ierr = MatDenseGetColumnVecWrite(D, 0, &cD);CHKERRQ(ierr);
          ierr = MatMult(A, cC, cD);CHKERRQ(ierr);
          ierr = PetscPrintf(PETSC_COMM_WORLD,"Benchmarking MatMult: with A %s %Dx%D\n",Atype,AM,AN);
          ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
          for (mm = 0; mm < trial; ++mm) {
            ierr = MatMult(A, cC, cD);CHKERRQ(ierr);
          }
          ierr = PetscLogStagePop();CHKERRQ(ierr);
          ierr = MatDenseRestoreColumnVecRead(C, 0, &cC);CHKERRQ(ierr);
          ierr = MatDenseRestoreColumnVecWrite(D, 0, &cD);CHKERRQ(ierr);
        } else {
          const PetscScalar *c_ptr;
          PetscScalar       *d_ptr;

          ierr = MatDenseGetArrayRead(C, &c_ptr);CHKERRQ(ierr);
          ierr = MatDenseGetArrayWrite(D, &d_ptr);CHKERRQ(ierr);
          ierr = PetscPrintf(PETSC_COMM_WORLD,"Benchmarking mkl_sparse_d_mv: with A %s %Dx%D\n",Atype,AM,AN);
          PetscStackCallMKLSparse(mkl_sparse_d_mv,(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, c_ptr, 0.0, d_ptr));
          ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
          for (mm = 0; mm < trial; ++mm) {
            PetscStackCallMKLSparse(mkl_sparse_d_mv,(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, c_ptr, 0.0, d_ptr));
          }
          ierr = PetscLogStagePop();CHKERRQ(ierr);
          ierr = MatDenseRestoreArrayWrite(D, &d_ptr);CHKERRQ(ierr);
          ierr = MatDenseRestoreArrayRead(C, &c_ptr);CHKERRQ(ierr);
        }

        if (check) {
          ierr = MatMatMultEqual(A,C,D,10,&flg);CHKERRQ(ierr);
          if (!flg) {
            MatType Dtype;

            ierr = MatGetType(D,&Dtype);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with A %s%s, C %s, D %s, Nk %D\n",Atype,mkl ? "mkl" : "",Ctype,Dtype,N[k]);CHKERRQ(ierr);
          }
        }

        /* MKL implementation seems buggy for ABt */
        /* A*Bt */
        if (!mkl && trans && N[k] > 1) {
          MatType Cttype;

          ierr = MatTranspose(C, MAT_INITIAL_MATRIX, &Ct);CHKERRQ(ierr);
          ierr = MatDestroy(&C);CHKERRQ(ierr);
          ierr = MatGetType(Ct,&Cttype);CHKERRQ(ierr);

          if (!mkl) {
            ierr = MatProductCreateWithMat(A, Ct, NULL, D);CHKERRQ(ierr);
            ierr = MatProductSetType(D, MATPRODUCT_ABt);CHKERRQ(ierr);
            ierr = MatProductSetFromOptions(D);CHKERRQ(ierr);
            ierr = MatProductSymbolic(D);CHKERRQ(ierr);
            ierr = MatProductNumeric(D);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD,"Benchmarking MatProduct %s: with A %s %Dx%D and Bt %s %Dx%D\n",MatProductTypes[MATPRODUCT_ABt],Atype,AM,AN,Cttype,CM,CN);
            ierr = PetscLogStagePush(tstage);CHKERRQ(ierr);
            for (mm = 0; mm < trial; ++mm) {
              ierr = MatProductNumeric(D);CHKERRQ(ierr);
            }
            ierr = PetscLogStagePop();CHKERRQ(ierr);
          } else {
            const PetscScalar *c_ptr;
            PetscScalar       *d_ptr;

            PetscStackCallMKLSparse(mkl_sparse_set_mm_hint,(spr,SPARSE_OPERATION_NON_TRANSPOSE, descr, SPARSE_LAYOUT_ROW_MAJOR, N[k], 1 + trial));
            PetscStackCallMKLSparse(mkl_sparse_optimize,(spr));
            ierr = MatDenseGetArrayRead(Ct, &c_ptr);CHKERRQ(ierr);
            ierr = MatDenseGetArrayWrite(D, &d_ptr);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD,"Benchmarking mkl_sparse_d_mm (ROW_MAJOR): with A %s %Dx%D and B %s %Dx%D\n",Atype,AM,AN,Cttype,CM,CN);
            PetscStackCallMKLSparse(mkl_sparse_d_mm,(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, SPARSE_LAYOUT_ROW_MAJOR, c_ptr, CN, CM, 0.0, d_ptr, CM));
            ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
            for (mm = 0; mm < trial; ++mm) {
              PetscStackCallMKLSparse(mkl_sparse_d_mm,(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, SPARSE_LAYOUT_ROW_MAJOR, c_ptr, CN, CM, 0.0, d_ptr, CM));
            }
            ierr = PetscLogStagePop();CHKERRQ(ierr);
            ierr = MatDenseRestoreArrayWrite(D, &d_ptr);CHKERRQ(ierr);
            ierr = MatDenseRestoreArrayRead(Ct, &c_ptr);CHKERRQ(ierr);
          }
        }

        if (Ct && check) {
          ierr = MatMatTransposeMultEqual(A,Ct,D,10,&flg);CHKERRQ(ierr);
          if (!flg) {
            MatType Dtype;
            ierr = MatGetType(D,&Dtype);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with A %s%s, C %s, D %s, Nk %D\n",Atype,mkl ? "mkl" : "",Ctype,Dtype,N[k]);CHKERRQ(ierr);
          }
        }
        ierr = MatDestroy(&Ct);CHKERRQ(ierr);
        ierr = MatDestroy(&C);CHKERRQ(ierr);
        ierr = MatDestroy(&D);CHKERRQ(ierr);
      }
      if (mkl) {
        PetscStackCallMKLSparse(mkl_sparse_destroy,(spr));
        ierr = PetscFree(ia_ptr);CHKERRQ(ierr);
        ierr = PetscFree(ja_ptr);CHKERRQ(ierr);
      }
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  for (m = 0; m < ntype; ++m) {
    ierr = PetscFree(type[m]);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return 0;
}